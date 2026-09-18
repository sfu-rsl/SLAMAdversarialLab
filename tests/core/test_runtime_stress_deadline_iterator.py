"""Tests for the runtime-stress DeadlineIterator."""

from __future__ import annotations

import itertools
import json
from typing import Iterator

import pytest

from slamadversariallab.runtime_stress.deadline_iterator import DeadlineIterator


@pytest.fixture
def fake_clock(monkeypatch):
    """Replace time.monotonic with a manually-advanced clock.

    Yields a callable ``advance(seconds)`` that bumps the fake clock.
    The DeadlineIterator reads time.monotonic on each __next__ call.

    ``time.sleep`` is also stubbed so the iterator's producer pacing
    advances the same fake clock instead of really sleeping. A manual
    ``advance(dt)`` between yields models the SLAM's per-frame compute
    time; a pacing ``sleep(dt)`` models the SLAM idling until the next
    frame is captured. Both move the one shared clock.
    """
    from slamadversariallab.runtime_stress import deadline_iterator

    current = [0.0]

    def fake_monotonic() -> float:
        return current[0]

    def fake_sleep(seconds: float) -> None:
        current[0] += seconds

    def advance(seconds: float) -> None:
        current[0] += seconds

    monkeypatch.setattr(deadline_iterator.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(deadline_iterator.time, "sleep", fake_sleep)
    return advance


def test_empty_list_stops_immediately(fake_clock):
    it = DeadlineIterator([], target_fps=30.0)
    with pytest.raises(StopIteration):
        next(it)
    assert it.survivors == []
    assert it.dropped == []


def test_invalid_target_fps_raises():
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=0)
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=-1.0)


def test_no_skip_when_consumer_keeps_up(fake_clock):
    """Consumer faster than target: every item yielded, no drops."""
    it = DeadlineIterator(["a", "b", "c", "d", "e"], target_fps=10.0)
    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.05)  # 50ms per frame, target period 100ms — well under

    assert yielded == ["a", "b", "c", "d", "e"]
    assert it.survivors == [0, 1, 2, 3, 4]
    assert it.dropped == []


def test_skip_when_consumer_is_slow(fake_clock):
    """Consumer at 200ms/frame on a 100ms-target: skip every other frame."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0)  # period = 100ms

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.200)  # 200ms per frame; double the period

    # Walking through:
    #   t=0,    target=0,  yield 0, advance 200ms
    #   t=200,  target=2,  skip 1, yield 2, advance 200ms
    #   t=400,  target=4,  skip 3, yield 4, advance 200ms
    #   t=600,  target=6,  skip 5, yield 6, advance 200ms
    #   t=800,  target=8,  skip 7, yield 8, advance 200ms
    #   t=1000, target=10, skip 9, exhausted.
    assert yielded == [0, 2, 4, 6, 8]
    assert it.survivors == [0, 2, 4, 6, 8]
    assert it.dropped == [1, 3, 5, 7, 9]


def test_clock_starts_on_first_next_not_construction(fake_clock):
    """Setup time before first __next__ shouldn't cost frame budget."""
    items = list(range(5))
    it = DeadlineIterator(items, target_fps=10.0)

    fake_clock(10.0)  # simulate 10s of model-loading before first next
    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.05)  # 50ms per frame after start

    assert yielded == [0, 1, 2, 3, 4]
    assert it.dropped == []


def test_writes_json_log_when_path_set(fake_clock, tmp_path, monkeypatch):
    log_path = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log_path))

    items = list(range(6))
    it = DeadlineIterator(items, target_fps=10.0)
    for _ in it:
        fake_clock(0.200)  # slow consumer

    assert log_path.exists()
    with open(log_path) as f:
        payload = json.load(f)
    assert payload["survivors"] == it.survivors
    assert payload["dropped"] == it.dropped
    assert payload["total_items"] == 6
    assert payload["target_fps"] == pytest.approx(10.0)


def test_no_log_written_when_path_unset(fake_clock, tmp_path, monkeypatch):
    monkeypatch.delenv("SAL_DROP_LOG_PATH", raising=False)
    it = DeadlineIterator([0, 1, 2], target_fps=10.0)
    for _ in it:
        fake_clock(0.05)
    assert not (tmp_path / "drops.json").exists()


def test_len_returns_total_items(fake_clock):
    it = DeadlineIterator(list(range(42)), target_fps=10.0)
    assert len(it) == 42


def test_iteration_after_exhaustion_keeps_raising(fake_clock):
    it = DeadlineIterator([0, 1], target_fps=10.0)
    list(it)
    with pytest.raises(StopIteration):
        next(it)
    with pytest.raises(StopIteration):
        next(it)


def test_works_with_enumerate_and_skips_correctly(fake_clock):
    """Verify the SLAM-side pattern: for t, item in enumerate(iterator)."""
    items = ["F0", "F1", "F2", "F3", "F4", "F5"]
    it = DeadlineIterator(items, target_fps=10.0)

    received: list[tuple[int, str]] = []
    for t, item in enumerate(it):
        received.append((t, item))
        fake_clock(0.200)  # slow consumer

    # SLAM's t counter is sequential; mapped items come from survivors.
    assert [t for t, _ in received] == [0, 1, 2]
    assert [item for _, item in received] == ["F0", "F2", "F4"]
    assert it.survivors == [0, 2, 4]
    assert it.dropped == [1, 3, 5]


def test_skip_to_end_of_list_terminates_cleanly(fake_clock):
    """Consumer is so slow the wall clock is past the last item."""
    items = list(range(5))
    it = DeadlineIterator(items, target_fps=10.0)

    yielded = []
    yielded.append(next(it))  # t=0, yields 0
    fake_clock(10.0)           # now 10s elapsed; period 100ms; target = 100
    with pytest.raises(StopIteration):
        next(it)

    # All non-yielded items get logged as dropped.
    assert it.survivors == [0]
    assert it.dropped == [1, 2, 3, 4]


def test_can_be_consumed_with_list(fake_clock):
    """Sanity: a consumer that keeps up yields every item via list().

    Under pacing, ``list(it)`` (no per-item compute) idles until each
    frame's arrival, so all frames are delivered and none are dropped.
    """
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0)
    out = list(itertools.islice(it, 10))
    assert out == items
    assert it.dropped == []


def test_invalid_warmup_frames_raises():
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=10.0, warmup_frames=-1)


def test_warmup_frames_bypasses_deadline(fake_clock):
    """The first warmup_frames items are yielded regardless of the wall clock.

    Use case: SLAM init takes 10 seconds (model load, CUDA JIT) on the
    first frame. Without warmup, that 10s consumes the deadline budget
    for ~100 subsequent frames at 10 FPS. With warmup_frames=5, those
    init costs are absorbed and the deadline schedule starts fresh
    after the SLAM is warmed up.
    """
    items = list(range(20))
    it = DeadlineIterator(items, target_fps=10.0, warmup_frames=5)

    yielded = []
    # The first frame is artificially slow (simulates CUDA JIT/init).
    yielded.append(next(it))
    fake_clock(10.0)  # init burns 10 s
    # Subsequent warmup frames are also un-deadlined.
    for _ in range(4):
        yielded.append(next(it))
        fake_clock(0.20)  # 200ms (slower than period) -- but warmup absorbs

    # All 5 warmup items yielded: 0..4
    assert yielded == [0, 1, 2, 3, 4]
    assert it.dropped == []  # warmup never drops
    assert it.survivors == [0, 1, 2, 3, 4]

    # Now we're past warmup. SLAM is warm; the clock starts on the next
    # __next__ call. Process at exactly the period (no drops expected).
    for _ in range(5):
        yielded.append(next(it))
        fake_clock(0.10)  # exactly target period

    # No drops; just continued sequential progress.
    assert yielded == list(range(10))
    assert it.dropped == []


def test_warmup_then_slow_consumer_drops_post_warmup(fake_clock):
    """After warmup ends, the deadline check applies. Slow consumer drops."""
    items = list(range(15))
    it = DeadlineIterator(items, target_fps=10.0, warmup_frames=3)

    yielded = []
    # Warmup: 3 frames with a slow init.
    for _ in range(3):
        yielded.append(next(it))
        fake_clock(5.0)  # very slow during warmup -- absorbed
    assert yielded == [0, 1, 2]
    assert it.dropped == []

    # Post-warmup: consumer at 200ms with 100ms period -> drop every other.
    for item in it:
        yielded.append(item)
        fake_clock(0.200)

    # Walking through post-warmup:
    #   t=0,    target=3,  yield 3, advance 200ms
    #   t=200,  target=5,  skip 4, yield 5, advance 200ms
    #   t=400,  target=7,  skip 6, yield 7, advance 200ms
    #   t=600,  target=9,  skip 8, yield 9, advance 200ms
    #   t=800,  target=11, skip 10, yield 11, advance 200ms
    #   t=1000, target=13, skip 12, yield 13, advance 200ms
    #   t=1200, target=15, skip 14, exhausted.
    assert yielded == [0, 1, 2, 3, 5, 7, 9, 11, 13]
    assert it.dropped == [4, 6, 8, 10, 12, 14]
    assert it.survivors == [0, 1, 2, 3, 5, 7, 9, 11, 13]


def test_warmup_zero_is_default_behavior(fake_clock):
    """warmup_frames=0 must be identical to old (pre-warmup) behavior."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0, warmup_frames=0)

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.200)

    assert yielded == [0, 2, 4, 6, 8]
    assert it.dropped == [1, 3, 5, 7, 9]


def test_warmup_larger_than_items_yields_all(fake_clock):
    """If warmup_frames >= len(items), all items are warmup, no drops possible."""
    items = list(range(5))
    it = DeadlineIterator(items, target_fps=10.0, warmup_frames=100)

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(10.0)  # arbitrarily slow

    assert yielded == [0, 1, 2, 3, 4]
    assert it.dropped == []


def test_warmup_frames_recorded_in_log(fake_clock, tmp_path, monkeypatch):
    log_path = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log_path))

    it = DeadlineIterator(list(range(5)), target_fps=10.0, warmup_frames=2)
    for _ in it:
        fake_clock(0.05)

    assert log_path.exists()
    with open(log_path) as f:
        payload = json.load(f)
    assert payload["warmup_frames"] == 2


# ---------------------------------------------------------------------------
# queue_size (bounded FIFO depth)
# ---------------------------------------------------------------------------


def test_invalid_queue_size_raises():
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=10.0, queue_size=0)
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=10.0, queue_size=-3)


def test_queue_size_one_is_default_behavior(fake_clock):
    """queue_size=1 must reproduce the original drop-every-other behavior."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0, queue_size=1)

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.200)  # 200ms per frame on a 100ms period

    # Identical to test_skip_when_consumer_is_slow (the default-queue case).
    assert yielded == [0, 2, 4, 6, 8]
    assert it.dropped == [1, 3, 5, 7, 9]


def test_larger_queue_drops_fewer_frames(fake_clock):
    """queue_size=2 buffers one extra frame, so the slow consumer drops less."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0, queue_size=2)

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.200)  # same slow consumer as the queue_size=1 case

    # Walking through (consume_target = arrived - 1):
    #   t=0,    arrived=0,  consume=-1, yield 0, advance 200ms
    #   t=200,  arrived=2,  consume=1,  yield 1, advance 200ms
    #   t=400,  arrived=4,  consume=3,  skip 2, yield 3, advance 200ms
    #   t=600,  arrived=6,  consume=5,  skip 4, yield 5, advance 200ms
    #   t=800,  arrived=8,  consume=7,  skip 6, yield 7, advance 200ms
    #   t=1000, arrived=10, consume=9,  skip 8, yield 9, advance 200ms
    #   t=1200, arrived=12, consume=11, exhausted.
    assert yielded == [0, 1, 3, 5, 7, 9]
    assert it.dropped == [2, 4, 6, 8]
    # Strictly fewer drops than queue_size=1 (which dropped 5).
    assert len(it.dropped) == 4


def test_queue_absorbs_stall_within_depth(fake_clock):
    """A transient stall shorter than the queue depth produces no drops."""
    items = list(range(8))
    it = DeadlineIterator(items, target_fps=10.0, queue_size=4)

    yielded = []
    yielded.append(next(it))  # frame 0
    fake_clock(0.300)         # 300ms stall (3 frames) -- within the 4-deep queue
    for item in it:
        yielded.append(item)
        fake_clock(0.100)     # back to exactly the period afterwards

    # The 4-deep buffer absorbs the 3-frame stall: nothing is dropped.
    assert yielded == list(range(8))
    assert it.dropped == []


def test_queue_size_recorded_in_log(fake_clock, tmp_path, monkeypatch):
    log_path = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log_path))

    it = DeadlineIterator(list(range(5)), target_fps=10.0, queue_size=3)
    for _ in it:
        fake_clock(0.05)

    assert log_path.exists()
    with open(log_path) as f:
        payload = json.load(f)
    assert payload["queue_size"] == 3


def test_writes_live_progress_file(fake_clock, tmp_path, monkeypatch):
    """Each yielded frame updates SAL_PROGRESS_PATH with the current index."""
    progress_path = tmp_path / "deadline_progress.json"
    monkeypatch.setenv("SAL_PROGRESS_PATH", str(progress_path))

    items = list(range(6))
    it = DeadlineIterator(items, target_fps=10.0)
    seen = []
    for _ in it:
        with open(progress_path) as f:
            seen.append(json.load(f)["frame"])
        fake_clock(0.05)  # keep up: every frame delivered

    # The progress file tracked the frame index as it advanced.
    assert seen == [0, 1, 2, 3, 4, 5]


def test_progress_file_records_skips(fake_clock, tmp_path, monkeypatch):
    """With a slow consumer the progress frame jumps past dropped frames."""
    progress_path = tmp_path / "deadline_progress.json"
    monkeypatch.setenv("SAL_PROGRESS_PATH", str(progress_path))

    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0)
    seen = []
    for _ in it:
        with open(progress_path) as f:
            seen.append(json.load(f)["frame"])
        fake_clock(0.200)  # drop every other frame

    assert seen == [0, 2, 4, 6, 8]


def test_no_progress_file_when_env_unset(fake_clock, tmp_path, monkeypatch):
    monkeypatch.delenv("SAL_PROGRESS_PATH", raising=False)
    it = DeadlineIterator([0, 1, 2], target_fps=10.0)
    for _ in it:
        fake_clock(0.05)
    assert not (tmp_path / "deadline_progress.json").exists()


# ---------------------------------------------------------------------------
# drop_policy (drop_oldest = keep-newest, drop_newest = tail-drop)
# ---------------------------------------------------------------------------


def test_invalid_drop_policy_raises():
    with pytest.raises(ValueError):
        DeadlineIterator([0, 1, 2], target_fps=10.0, drop_policy="nonsense")


def test_drop_policy_default_is_drop_oldest(fake_clock):
    """Default policy must reproduce the keep-newest behavior exactly."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0, queue_size=2)  # default policy
    for _ in it:
        fake_clock(0.200)
    assert it.survivors == [0, 1, 3, 5, 7, 9]
    assert it.dropped == [2, 4, 6, 8]


def test_drop_newest_keeps_oldest_drops_incoming(fake_clock):
    """Tail-drop keeps the buffered (older) frames and rejects new arrivals."""
    items = list(range(10))
    it = DeadlineIterator(
        items, target_fps=10.0, queue_size=2, drop_policy="drop_newest"
    )
    for _ in it:
        fake_clock(0.200)  # same slow consumer as the drop_oldest case

    # Tail-drop grinds through the older contiguous backlog (0,1,2,3) before
    # it starts shedding, and the dropped frames are the *incoming* ones.
    assert it.survivors == [0, 1, 2, 3, 5, 7, 9]
    assert it.dropped == [4, 6, 8]
    # Strictly different survivor set from drop_oldest (which keeps F3 not F2).
    assert it.survivors != [0, 1, 3, 5, 7, 9]


def test_drop_newest_fast_consumer_has_no_drops(fake_clock):
    """A consumer that keeps up never overflows the buffer, so nothing drops."""
    items = list(range(6))
    it = DeadlineIterator(
        items, target_fps=10.0, queue_size=2, drop_policy="drop_newest"
    )
    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.05)  # 50ms < 100ms period: SLAM faster than camera

    assert yielded == items
    assert it.dropped == []


def test_drop_policy_recorded_in_log(fake_clock, tmp_path, monkeypatch):
    log_path = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log_path))

    it = DeadlineIterator(
        list(range(5)), target_fps=10.0, drop_policy="drop_newest"
    )
    for _ in it:
        fake_clock(0.05)

    assert log_path.exists()
    with open(log_path) as f:
        payload = json.load(f)
    assert payload["drop_policy"] == "drop_newest"


# ---------------------------------------------------------------------------
# producer pacing (a fast SLAM cannot outrun the source frame rate)
# ---------------------------------------------------------------------------


def test_fast_consumer_is_paced_to_source_rate(monkeypatch):
    """A SLAM faster than target_fps must wait for each frame's arrival."""
    from slamadversariallab.runtime_stress import deadline_iterator

    clock = [0.0]
    sleeps: list[float] = []
    monkeypatch.setattr(deadline_iterator.time, "monotonic", lambda: clock[0])

    def rec_sleep(dt: float) -> None:
        sleeps.append(dt)
        clock[0] += dt

    monkeypatch.setattr(deadline_iterator.time, "sleep", rec_sleep)

    items = list(range(5))
    it = DeadlineIterator(items, target_fps=10.0)  # period 100ms

    yielded = []
    for item in it:
        yielded.append(item)
        clock[0] += 0.02  # SLAM computes each frame in 20ms (way under 100ms)

    # Fast SLAM keeps every frame -- nothing is late.
    assert yielded == items
    assert it.dropped == []
    # ...but it was paced: it slept before each post-first frame instead of
    # racing ahead, and never processed faster than the 100ms source period.
    assert len(sleeps) == 4
    assert all(s == pytest.approx(0.08) for s in sleeps)


def test_pacing_does_not_rescue_a_slow_consumer(fake_clock):
    """Pacing only slows down a fast SLAM; a slow one still drops frames."""
    items = list(range(10))
    it = DeadlineIterator(items, target_fps=10.0)  # period 100ms

    yielded = []
    for item in it:
        yielded.append(item)
        fake_clock(0.200)  # 200ms compute -- always behind, never paced

    assert yielded == [0, 2, 4, 6, 8]
    assert it.dropped == [1, 3, 5, 7, 9]


def test_paced_total_time_at_least_stream_duration(monkeypatch):
    """Total elapsed for a fast SLAM must be >= the real stream duration."""
    from slamadversariallab.runtime_stress import deadline_iterator

    clock = [0.0]
    monkeypatch.setattr(deadline_iterator.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        deadline_iterator.time, "sleep", lambda dt: clock.__setitem__(0, clock[0] + dt)
    )

    items = list(range(20))
    it = DeadlineIterator(items, target_fps=10.0)  # 20 frames @ 10fps => 1.9s span
    for _ in it:
        clock[0] += 0.001  # 1ms compute -- absurdly fast

    # Frame 19 arrives at (19) * 0.1 = 1.9s; the run cannot finish sooner.
    assert clock[0] >= 1.9
