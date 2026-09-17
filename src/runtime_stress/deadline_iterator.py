"""Deadline-aware iterator for offline SLAM runs.

Wraps a list of frame items so that, at iteration time, items whose
wall-clock deadline has passed are silently skipped. Models a real-time
deployment where late frames get dropped before the SLAM ever sees them.

It also paces the stream to the source frame rate: a SLAM faster than
``target_fps`` is made to wait for each frame's arrival time, since a real
camera cannot be outrun. Pacing plus a bounded queue (``queue_size``)
together model the producer / bounded-buffer / consumer pipeline of a
real robot: frames are produced at a fixed cadence, buffered up to
``queue_size`` deep, and dropped only when the SLAM falls far enough
behind that the buffer overflows.

Designed to be imported from inside a SLAM's own Python entry point
(e.g. DROID-SLAM's ``demo.py``) when the ``SAL_DEADLINE_FPS`` env var
is set. On iterator exhaustion, writes a JSON drop log to
``SAL_DROP_LOG_PATH`` so the framework can align trajectory output
with dataset indices for ATE/RPE evaluation.

Intentionally has no SAL package imports so it can be loaded via raw
``sys.path`` injection from a SLAM subprocess in any conda env.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Deque, Iterable, Iterator, List, Optional


class DeadlineIterator:
    """Iterator that skips items whose wall-clock deadline has passed.

    The wall clock starts on the first post-warmup ``__next__`` call,
    not on construction, so SLAM model-load and GPU-init don't eat
    into the frame-rate budget.

    Parameters
    ----------
    items:
        The original list of items (typically file paths, indices, or
        any per-frame value the SLAM iterates over). Materialized into
        a list internally so the iterator can index into it when
        skipping ahead.
    target_fps:
        Target frame rate in frames per second. Post-warmup, item
        ``warmup_frames + k`` has deadline ``k / target_fps`` seconds
        from the moment warmup ends.
    warmup_frames:
        Number of items at the start of the iteration that bypass the
        deadline check entirely. Use this to absorb per-SLAM
        initialization stalls (CUDA kernel JIT, model load, lazy
        allocator setup) that would otherwise look like late frames
        even though they don't represent steady-state compute. The
        wall clock starts on the first ``__next__`` call after the
        warmup phase ends. Default is 0 (no warmup, clock starts on
        the first call).
    queue_size:
        Depth of the bounded FIFO buffer between the (simulated) camera
        and the SLAM, modeling a ROS-style ``queue_size``. Frames arrive
        at ``target_fps`` and pile up to ``queue_size`` deep; the SLAM
        consumes the oldest frame still buffered. Only when the buffer
        overflows is the oldest waiting frame dropped. Default is 1,
        which reproduces the original "always process the freshest
        arrival, drop everything older" behavior (a 1-deep buffer). A
        larger queue absorbs transient compute stalls (fewer drops) at
        the cost of the SLAM processing staler frames (latency up to
        ``queue_size - 1`` frames).
    drop_policy:
        What happens when the buffer is full and a new frame arrives:

        * ``"drop_oldest"`` (default): evict the oldest waiting frame and
          keep the newest ``queue_size`` (a ROS-style ring buffer). The
          SLAM stays close to real time. This is computable from the
          clock alone, so it uses no stored buffer.
        * ``"drop_newest"``: reject the incoming frame and keep the
          frames already buffered (tail-drop). The SLAM grinds through a
          contiguous backlog of older frames. This is path-dependent
          (what's buffered depends on the admission history), so it is
          simulated with a real FIFO ``deque``.

    Notes
    -----
    Skipped items are not loaded from disk. Calling code that does
    expensive work per item (``cv2.imread``, GPU upload) only pays for
    items the iterator actually yields.
    """

    _DROP_POLICIES = ("drop_oldest", "drop_newest")

    def __init__(
        self,
        items: Iterable,
        target_fps: float,
        warmup_frames: int = 0,
        queue_size: int = 1,
        drop_policy: str = "drop_oldest",
    ) -> None:
        if target_fps <= 0:
            raise ValueError(f"target_fps must be positive, got {target_fps}")
        if warmup_frames < 0:
            raise ValueError(
                f"warmup_frames must be non-negative, got {warmup_frames}"
            )
        if queue_size < 1:
            raise ValueError(f"queue_size must be >= 1, got {queue_size}")
        if drop_policy not in self._DROP_POLICIES:
            raise ValueError(
                f"drop_policy must be one of {self._DROP_POLICIES}, got {drop_policy!r}"
            )
        self.items: List = list(items)
        self.period: float = 1.0 / float(target_fps)
        self.warmup_frames: int = int(warmup_frames)
        self.queue_size: int = int(queue_size)
        self.drop_policy: str = drop_policy
        self.start: Optional[float] = None
        self.next_idx: int = 0
        self.dropped: List[int] = []
        self.survivors: List[int] = []
        # Per-handoff timing: [idx, entry_s, yield_s] relative to the first
        # __next__ entry. entry is when the SLAM asked for a frame (i.e. when
        # it finished the previous one), yield is when the frame was handed
        # over, so SLAM processing time for handoff k is
        # entry(k+1) - yield(k): pacing sleeps happen between entry and yield
        # and are excluded by construction. Costs two clock reads per frame.
        self.handoffs: List[List[float]] = []
        self._t0: Optional[float] = None
        self._end_entry: Optional[float] = None
        self._log_written: bool = False
        # drop_newest only: the actual FIFO buffer of admitted-but-unconsumed
        # frame indices, and the camera pointer (next index to "arrive").
        self._buffer: Deque[int] = deque()
        self._next_arrival: int = self.warmup_frames

    def __iter__(self) -> Iterator:
        return self

    def __next__(self):
        t_entry = time.monotonic()
        if self._t0 is None:
            self._t0 = t_entry

        # Warmup phase: yield items un-deadlined so SLAM init costs
        # don't count against the frame-rate budget.
        if self.next_idx < self.warmup_frames:
            if self.next_idx >= len(self.items):
                self._end_entry = round(t_entry - self._t0, 6)
                self._save_log()
                raise StopIteration
            item = self.items[self.next_idx]
            self.survivors.append(self.next_idx)
            self._write_progress(self.next_idx)
            self._record_handoff(self.next_idx, t_entry)
            self.next_idx += 1
            return item

        # First post-warmup call: start the wall clock.
        if self.start is None:
            self.start = time.monotonic()

        # drop_newest needs a real FIFO buffer (path-dependent), so it runs
        # a separate simulation. drop_oldest stays the clock-only arithmetic.
        if self.drop_policy == "drop_newest":
            return self._next_drop_newest(t_entry)

        # Nothing left to deliver (the previous call consumed the last
        # frame). Check before pacing so we don't sleep for a frame that
        # doesn't exist.
        if self.next_idx >= len(self.items):
            self._end_entry = round(t_entry - self._t0, 6)
            self._save_log()
            raise StopIteration

        # Producer pacing: a real camera produces frame ``next_idx`` at a
        # fixed cadence, so the SLAM cannot consume it any earlier than its
        # arrival time, no matter how fast the SLAM is. If we're ahead of
        # schedule (the SLAM is faster than the source rate), block until
        # the frame would have been captured. This stops a fast SLAM from
        # processing the stream faster than real-time and is what lets the
        # bounded queue actually fill under bursty load.
        arrival = self.start + (self.next_idx - self.warmup_frames) * self.period
        now = time.monotonic()
        if now < arrival:
            time.sleep(arrival - now)

        elapsed = time.monotonic() - self.start
        # Newest frame that has "arrived" by now. Frames are numbered
        # relative to warmup_frames: at elapsed=k*period the newest
        # arrival is item warmup_frames + k. The 1e-9 epsilon absorbs
        # floating-point roundoff so that an elapsed of "exactly k*period"
        # that lost precision in subtraction (e.g. 15.2 - 15.0 = 0.1999...
        # due to IEEE-754 representation) still resolves to k.
        arrived = int(elapsed / self.period + 1e-9) + self.warmup_frames

        # Bounded FIFO of depth queue_size: a frame may wait in the buffer
        # for queue_size-1 newer arrivals before its deadline passes, so
        # the oldest still-live frame is queue_size-1 behind the newest
        # arrival. Anything older expired (was evicted) and is dropped.
        # queue_size == 1 reduces this to the original "consume the newest
        # arrival, drop everything older" behavior.
        consume_target = arrived - (self.queue_size - 1)

        if consume_target > self.next_idx:
            self.dropped.extend(
                range(self.next_idx, min(consume_target, len(self.items)))
            )
            self.next_idx = consume_target

        if self.next_idx >= len(self.items):
            self._end_entry = round(t_entry - self._t0, 6)
            self._save_log()
            raise StopIteration

        item = self.items[self.next_idx]
        self.survivors.append(self.next_idx)
        self._write_progress(self.next_idx)
        self._record_handoff(self.next_idx, t_entry)
        self.next_idx += 1
        return item

    def _admit_arrivals(self, arrived: int) -> None:
        """Push camera frames that have arrived by now into the FIFO buffer.

        Used only by the ``drop_newest`` path. Frames from ``_next_arrival``
        up to ``arrived`` enter the buffer in order while there is room; once
        the buffer is full the incoming frame is rejected (tail-drop) and
        recorded as dropped. Never admits a frame past the end of the stream.
        """
        last = len(self.items) - 1
        while self._next_arrival <= arrived and self._next_arrival <= last:
            if len(self._buffer) < self.queue_size:
                self._buffer.append(self._next_arrival)
            else:
                self.dropped.append(self._next_arrival)  # tail-drop: reject incoming
            self._next_arrival += 1

    def _record_handoff(self, idx: int, t_entry: float) -> None:
        self.handoffs.append(
            [idx, round(t_entry - self._t0, 6),
             round(time.monotonic() - self._t0, 6)]
        )

    def _next_drop_newest(self, t_entry: float):
        """Deliver the next frame under the tail-drop (keep-existing) policy.

        Frames arrive at the source rate into a bounded FIFO; the SLAM
        consumes the oldest buffered frame. When the SLAM is faster than the
        camera the buffer empties and we pace (wait) for the next arrival,
        mirroring the drop_oldest path's producer pacing.
        """
        elapsed = time.monotonic() - self.start
        arrived = int(elapsed / self.period + 1e-9) + self.warmup_frames
        self._admit_arrivals(arrived)

        # Buffer empty: the SLAM has drained everything available and must
        # wait for the next frame to be captured (it can't outrun the camera).
        while not self._buffer:
            if self._next_arrival >= len(self.items):
                self._end_entry = round(t_entry - self._t0, 6)
                self._save_log()
                raise StopIteration
            arrival = self.start + (self._next_arrival - self.warmup_frames) * self.period
            now = time.monotonic()
            if now < arrival:
                time.sleep(arrival - now)
            elapsed = time.monotonic() - self.start
            arrived = int(elapsed / self.period + 1e-9) + self.warmup_frames
            self._admit_arrivals(arrived)

        frame = self._buffer.popleft()
        self.survivors.append(frame)
        self._write_progress(frame)
        self._record_handoff(frame, t_entry)
        return self.items[frame]

    def __len__(self) -> int:
        return len(self.items)

    def _write_progress(self, frame_index: int) -> None:
        """Publish the current sampled-frame index for frame-anchored phases.

        Written to ``SAL_PROGRESS_PATH`` (a file in the SLAM's bind-mounted
        output dir) so the host orchestrator can switch control phases when
        the SLAM reaches a target frame. Atomic (temp + rename) so a reader
        never sees a half-written file. No-op when the env var is unset, and
        never fatal: a logging failure must not crash the SLAM.
        """
        path = os.environ.get("SAL_PROGRESS_PATH")
        if not path:
            return
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "frame": frame_index,
                        "survivors": len(self.survivors),
                        "dropped": len(self.dropped),
                    },
                    f,
                )
            os.replace(tmp, path)
        except OSError:
            pass

    def _save_log(self) -> None:
        if self._log_written:
            return
        path = os.environ.get("SAL_DROP_LOG_PATH")
        if not path:
            self._log_written = True
            return
        payload = {
            "survivors": self.survivors,
            "dropped": self.dropped,
            "target_fps": 1.0 / self.period,
            "total_items": len(self.items),
            "warmup_frames": self.warmup_frames,
            "queue_size": self.queue_size,
            "drop_policy": self.drop_policy,
            "handoffs": self.handoffs,
            "end_entry_s": self._end_entry,
        }
        try:
            with open(path, "w") as f:
                json.dump(payload, f)
            self._log_written = True
        except OSError as exc:
            # Loud breadcrumb, not a silent swallow. A missing drop log
            # misaligns counter-keyed SLAMs (DROID/DPVO); their entry scripts
            # and host wrappers now RAISE on a missing log, so surface the root
            # cause here on the SLAM's stderr (captured in slam_output.log)
            # instead of hiding it. Not re-raised so a log-write failure does
            # not crash timestamp-keyed SLAMs that do not depend on the log.
            import sys as _sys
            print(
                f"[SAL][deadline_iterator] ERROR: failed to write drop log to "
                f"{path!r}: {exc}. Counter-keyed SLAMs (DROID/DPVO) will now "
                f"fail their trajectory conversion by design.",
                file=_sys.stderr,
                flush=True,
            )
