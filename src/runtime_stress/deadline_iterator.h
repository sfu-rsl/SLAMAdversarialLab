// Deadline-aware frame selector for offline C++ SLAM runs.
//
// C++ port of ``src/runtime_stress/deadline_iterator.py``. Same job: at
// iteration time, drop frames whose wall-clock deadline has passed, modeling a
// real-time deployment where late frames get discarded before the SLAM sees
// them. It also paces the stream to the source rate, since a real camera cannot
// be outrun, and supports a bounded queue with two drop policies.
//
// Designed to be ``#include``-d from a C++ SLAM's own KITTI/TUM/EuRoC entry
// point (e.g. ORB-SLAM3's ``mono_kitti.cc``) when the ``SAL_DEADLINE_FPS`` env
// var is set. On exhaustion it writes the same JSON drop log to
// ``SAL_DROP_LOG_PATH`` that the Python iterator writes, so the framework can
// report drop_rate identically for Python and C++ SLAMs.
//
// CONTRACT SHARED WITH deadline_iterator.py -- KEEP THE TWO IN SYNC:
//   * env vars:  SAL_DEADLINE_FPS, SAL_DEADLINE_WARMUP_FRAMES,
//                SAL_DEADLINE_QUEUE_SIZE, SAL_DEADLINE_DROP_POLICY,
//                SAL_DROP_LOG_PATH, SAL_PROGRESS_PATH
//   * drop-log JSON schema: {"survivors":[int],"dropped":[int],"target_fps":
//     float,"total_items":int,"warmup_frames":int,"queue_size":int,
//     "drop_policy":str,"handoffs":[[idx,entry_s,yield_s]],"end_entry_s":float}
//     handoffs: per-delivered-frame timing relative to the first next() entry;
//     entry is when the SLAM asked (finished the previous frame), yield when
//     the frame was handed over, so per-frame SLAM processing time is
//     entry(k+1) - yield(k) and pacing sleeps are excluded by construction.
//   * drop semantics: drop_oldest is clock-only; drop_newest uses a real FIFO.
//
// Unlike the Python iterator, this works over frame INDICES [0, n) rather than
// the items themselves: the C++ caller owns the frame list and does its own
// per-frame imread, so the iterator only needs to decide which indices survive.
// It is consumed lazily and interleaved with the real ``Track...`` call -- that
// interleaving is what makes the drop decision faithful (a frame is dropped
// only because the actual SLAM was too slow to meet its deadline).
//
// Header-only; depends only on the C++ stdlib and <unistd.h>.

#ifndef SAL_DEADLINE_ITERATOR_H
#define SAL_DEADLINE_ITERATOR_H

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace sal {

class DeadlineIterator {
 public:
  // total_items: number of frames the caller will iterate over (0..n-1).
  // Reads the SAL_DEADLINE_* env vars. When SAL_DEADLINE_FPS is unset/empty the
  // iterator is inactive (active() == false) and the caller should run its
  // normal loop. Throws std::invalid_argument on a present-but-invalid config
  // (fail fast, mirroring the Python constructor's ValueError checks).
  explicit DeadlineIterator(int total_items)
      : n_(total_items < 0 ? 0 : total_items) {
    const char* fps_env = std::getenv("SAL_DEADLINE_FPS");
    if (fps_env == nullptr || fps_env[0] == '\0') {
      active_ = false;  // harness off; behaves as a no-op
      return;
    }
    active_ = true;
    fps_ = parse_double(fps_env);
    if (fps_ <= 0.0) {
      throw std::invalid_argument("SAL_DEADLINE_FPS must be positive");
    }
    period_ = 1.0 / fps_;
    warmup_ = getenv_int("SAL_DEADLINE_WARMUP_FRAMES", 0);
    if (warmup_ < 0) {
      throw std::invalid_argument("SAL_DEADLINE_WARMUP_FRAMES must be >= 0");
    }
    queue_ = getenv_int("SAL_DEADLINE_QUEUE_SIZE", 1);
    if (queue_ < 1) {
      throw std::invalid_argument("SAL_DEADLINE_QUEUE_SIZE must be >= 1");
    }
    const char* policy_env = std::getenv("SAL_DEADLINE_DROP_POLICY");
    policy_ = (policy_env != nullptr && policy_env[0] != '\0') ? policy_env
                                                               : "drop_oldest";
    if (policy_ != "drop_oldest" && policy_ != "drop_newest") {
      throw std::invalid_argument(
          "SAL_DEADLINE_DROP_POLICY must be drop_oldest or drop_newest");
    }
    next_arrival_ = warmup_;
  }

  // Write the drop log on destruction if the consumer never reached
  // exhaustion (e.g. a callback-driven reader whose stream ended early).
  // write_log() is idempotent, so explicit calls remain safe.
  ~DeadlineIterator() {
    if (active_) {
      write_log();
    }
  }

  // True when the deadline harness is active for this run.
  bool active() const { return active_; }

  // Push-style variant of next() for callback-driven readers (e.g. OKVIS's
  // DatasetReader) that iterate frames themselves and ask, per frame k
  // (monotonically increasing from 0), whether to deliver it to the SLAM.
  // Semantics are identical to the pull loop `for(k=next(); k>=0; k=next())`:
  // paces to the survivor's arrival time and drops frames whose deadline
  // passed because the (blocking) consumer was too slow. Returns true when
  // frame k must be delivered, false when it was dropped (or the deadline
  // stream is exhausted). Always true when inactive.
  bool should_deliver(int k) {
    if (!active_) {
      return true;
    }
    if (!pending_primed_) {
      pending_survivor_ = next();
      pending_primed_ = true;
    }
    if (pending_survivor_ < 0) {
      return false;  // deadline stream exhausted
    }
    if (k < pending_survivor_) {
      return false;  // frame k was skipped by the drop policy
    }
    // k == pending_survivor_ (k > pending cannot happen with monotone k).
    pending_survivor_ = next();
    return true;
  }

  // Returns the next frame index to process, or -1 when the stream is
  // exhausted. Records the returned index as a survivor and publishes progress.
  // Paces to the frame's arrival time (so a fast SLAM cannot outrun the camera)
  // and applies the drop policy. Must be called once per consumed frame,
  // interleaved with the real per-frame work.
  int next() {
    double t_entry = mono_now();
    if (!t0_set_) {
      t0_ = t_entry;
      t0_set_ = true;
    }

    // Warmup phase: yield un-deadlined so SLAM init costs don't count against
    // the frame-rate budget.
    if (next_idx_ < warmup_) {
      if (next_idx_ >= n_) {
        set_end_entry(t_entry);
        write_log();
        return -1;
      }
      int idx = next_idx_;
      survivors_.push_back(idx);
      write_progress(idx);
      record_handoff(idx, t_entry);
      next_idx_ += 1;
      return idx;
    }

    // First post-warmup call: start the wall clock.
    if (!clock_started_) {
      start_clock();
      clock_started_ = true;
    }

    if (policy_ == "drop_newest") {
      return next_drop_newest(t_entry);
    }

    // drop_oldest: clock-only arithmetic.
    if (next_idx_ >= n_) {
      set_end_entry(t_entry);
      write_log();
      return -1;
    }

    // Producer pacing: block until frame next_idx_ would have been captured.
    double arrival = static_cast<double>(next_idx_ - warmup_) * period_;
    double now = elapsed();
    if (now < arrival) {
      sleep_seconds(arrival - now);
    }

    double el = elapsed();
    // Newest frame that has "arrived" by now (1e-9 epsilon absorbs roundoff,
    // matching the Python implementation).
    int arrived = static_cast<int>(el / period_ + 1e-9) + warmup_;
    // Bounded FIFO of depth queue_: the oldest still-live frame is queue_-1
    // behind the newest arrival; anything older expired and is dropped.
    int consume_target = arrived - (queue_ - 1);
    if (consume_target > next_idx_) {
      int stop = std::min(consume_target, n_);
      for (int k = next_idx_; k < stop; ++k) {
        dropped_.push_back(k);
      }
      next_idx_ = consume_target;
    }

    if (next_idx_ >= n_) {
      set_end_entry(t_entry);
      write_log();
      return -1;
    }

    int idx = next_idx_;
    survivors_.push_back(idx);
    write_progress(idx);
    record_handoff(idx, t_entry);
    next_idx_ += 1;
    return idx;
  }

  // Write the drop log to SAL_DROP_LOG_PATH. Idempotent and never fatal; a
  // no-op when the env var is unset. next() calls this on exhaustion; callers
  // should also call it after the loop to cover an early break.
  void write_log() {
    if (log_written_) {
      return;
    }
    const char* path = std::getenv("SAL_DROP_LOG_PATH");
    if (path == nullptr || path[0] == '\0') {
      log_written_ = true;
      return;
    }
    FILE* f = std::fopen(path, "w");
    if (f == nullptr) {
      // Loud breadcrumb, not a silent swallow (mirrors the Python iterator).
      // A missing drop log misaligns counter-keyed SLAMs; timestamp-keyed
      // C++ SLAMs (ORB-SLAM3, OKVIS2-X) do not depend on it, so we do not
      // abort here -- but we never hide the failure.
      std::fprintf(stderr,
                   "[SAL][deadline_iterator] ERROR: failed to open drop log "
                   "'%s' for writing.\n", path);
      return;
    }
    std::fputs("{\"survivors\":", f);
    write_int_array(f, survivors_);
    std::fputs(",\"dropped\":", f);
    write_int_array(f, dropped_);
    std::fprintf(f, ",\"target_fps\":%.10g", fps_);
    std::fprintf(f, ",\"total_items\":%d", n_);
    std::fprintf(f, ",\"warmup_frames\":%d", warmup_);
    std::fprintf(f, ",\"queue_size\":%d", queue_);
    std::fprintf(f, ",\"drop_policy\":\"%s\"", policy_.c_str());
    std::fputs(",\"handoffs\":[", f);
    for (size_t i = 0; i < handoffs_.size(); ++i) {
      if (i != 0) {
        std::fputc(',', f);
      }
      std::fprintf(f, "[%d,%.6f,%.6f]", handoffs_[i].idx, handoffs_[i].entry,
                   handoffs_[i].yld);
    }
    std::fputc(']', f);
    if (have_end_) {
      std::fprintf(f, ",\"end_entry_s\":%.6f}", end_entry_);
    } else {
      std::fputs(",\"end_entry_s\":null}", f);
    }
    std::fclose(f);
    log_written_ = true;
  }

#ifdef SAL_DEADLINE_TEST_CLOCK
  // Test-only hooks (present only in the -DSAL_DEADLINE_TEST_CLOCK build) for a
  // differential harness to drive the virtual clock: reset before a trace, then
  // advance by the modeled per-frame consumer cost between next() calls.
  static void sal_reset_test_clock() { test_clock() = 0.0; }
  static void sal_advance_test_clock(double seconds) { test_clock() += seconds; }
  const std::vector<int>& sal_survivors() const { return survivors_; }
  const std::vector<int>& sal_dropped() const { return dropped_; }
#endif

 private:
  struct Handoff {
    int idx;
    double entry;
    double yld;
  };

  void record_handoff(int idx, double t_entry) {
    handoffs_.push_back({idx, t_entry - t0_, mono_now() - t0_});
  }

  void set_end_entry(double t_entry) {
    if (!t0_set_) {  // exhausted before any frame: anchor at this entry
      t0_ = t_entry;
      t0_set_ = true;
    }
    end_entry_ = t_entry - t0_;
    have_end_ = true;
  }

  // drop_newest: deliver the oldest buffered frame (tail-drop on overflow).
  int next_drop_newest(double t_entry) {
    double el = elapsed();
    int arrived = static_cast<int>(el / period_ + 1e-9) + warmup_;
    admit_arrivals(arrived);

    while (buffer_.empty()) {
      if (next_arrival_ >= n_) {
        set_end_entry(t_entry);
        write_log();
        return -1;
      }
      double arrival = static_cast<double>(next_arrival_ - warmup_) * period_;
      double now = elapsed();
      if (now < arrival) {
        sleep_seconds(arrival - now);
      }
      el = elapsed();
      arrived = static_cast<int>(el / period_ + 1e-9) + warmup_;
      admit_arrivals(arrived);
    }

    int frame = buffer_.front();
    buffer_.pop_front();
    survivors_.push_back(frame);
    write_progress(frame);
    record_handoff(frame, t_entry);
    return frame;
  }

  // Push camera frames that have arrived into the bounded FIFO; tail-drop the
  // incoming frame when the buffer is full.
  void admit_arrivals(int arrived) {
    int last = n_ - 1;
    while (next_arrival_ <= arrived && next_arrival_ <= last) {
      if (static_cast<int>(buffer_.size()) < queue_) {
        buffer_.push_back(next_arrival_);
      } else {
        dropped_.push_back(next_arrival_);  // tail-drop: reject incoming
      }
      next_arrival_ += 1;
    }
  }

  // Publish the current sampled-frame index for frame-anchored phases. Atomic
  // (temp + rename) so a reader never sees a half-written file. No-op when the
  // env var is unset, and never fatal.
  void write_progress(int frame_index) {
    const char* path = std::getenv("SAL_PROGRESS_PATH");
    if (path == nullptr || path[0] == '\0') {
      return;
    }
    std::string tmp = std::string(path) + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) {
      return;
    }
    std::fprintf(f, "{\"frame\": %d, \"survivors\": %zu, \"dropped\": %zu}",
                 frame_index, survivors_.size(), dropped_.size());
    std::fclose(f);
    std::rename(tmp.c_str(), path);  // atomic replace; ignore failure
  }

  // --- Clock seam ---
  // Production reads the real monotonic clock. A test build
  // (-DSAL_DEADLINE_TEST_CLOCK) swaps in a deterministic virtual clock so a
  // differential harness can inject the exact same timing into the C++ and
  // Python iterators and compare their drop sets bit-for-bit. Production
  // behavior is unchanged: the #else branch is the original steady_clock /
  // usleep code, byte for byte.
#ifdef SAL_DEADLINE_TEST_CLOCK
  // Process-global virtual time in seconds. sleep_seconds advances it; the
  // harness advances it to model per-frame consumer cost between next() calls.
  static double& test_clock() {
    static double t = 0.0;
    return t;
  }
  void start_clock() { start_s_ = test_clock(); }
  double elapsed() const { return test_clock() - start_s_; }
  static void sleep_seconds(double seconds) {
    if (seconds > 0.0) {
      test_clock() += seconds;
    }
  }
#else
  void start_clock() { start_ = std::chrono::steady_clock::now(); }
  double elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start_)
        .count();
  }
  static void sleep_seconds(double seconds) {
    if (seconds > 0.0) {
      ::usleep(static_cast<useconds_t>(seconds * 1e6));
    }
  }
#endif

  // Absolute monotonic seconds on the same seam as elapsed(): the virtual
  // clock under the test build, steady_clock otherwise. Used only for the
  // handoff telemetry, never for drop decisions.
#ifdef SAL_DEADLINE_TEST_CLOCK
  double mono_now() const { return test_clock(); }
#else
  double mono_now() const {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
#endif

  static void write_int_array(FILE* f, const std::vector<int>& v) {
    std::fputc('[', f);
    for (size_t i = 0; i < v.size(); ++i) {
      if (i != 0) {
        std::fputc(',', f);
      }
      std::fprintf(f, "%d", v[i]);
    }
    std::fputc(']', f);
  }

  static double parse_double(const char* s) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s) {
      throw std::invalid_argument("invalid numeric SAL_DEADLINE env value");
    }
    return v;
  }

  static int getenv_int(const char* name, int default_value) {
    const char* s = std::getenv(name);
    if (s == nullptr || s[0] == '\0') {
      return default_value;
    }
    char* end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (end == s) {
      throw std::invalid_argument(std::string("invalid integer ") + name);
    }
    return static_cast<int>(v);
  }

  int n_ = 0;
  bool active_ = false;
  double fps_ = 0.0;
  double period_ = 0.0;
  int warmup_ = 0;
  int queue_ = 1;
  std::string policy_ = "drop_oldest";

  bool clock_started_ = false;
#ifdef SAL_DEADLINE_TEST_CLOCK
  double start_s_ = 0.0;  // virtual-clock start time (seconds)
#else
  std::chrono::steady_clock::time_point start_{};
#endif
  // should_deliver() state: the survivor index awaiting delivery.
  bool pending_primed_ = false;
  int pending_survivor_ = -1;
  int next_idx_ = 0;
  int next_arrival_ = 0;  // drop_newest: next camera frame to "arrive"
  std::deque<int> buffer_;
  std::vector<int> survivors_;
  std::vector<int> dropped_;
  std::vector<Handoff> handoffs_;
  bool t0_set_ = false;
  double t0_ = 0.0;
  double end_entry_ = 0.0;
  bool have_end_ = false;
  bool log_written_ = false;
};

}  // namespace sal

#endif  // SAL_DEADLINE_ITERATOR_H
