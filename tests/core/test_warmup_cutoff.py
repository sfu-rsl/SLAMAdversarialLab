"""Tests for the deadline warmup-cutoff timestamp computation.

The cutoff must be stride-aware: warmup_frames counts stride-sampled frames
(the frames the SLAM actually processes), so on a stride-2 SLAM (DROID-SLAM,
DPVO on TUM) sampled frame ``warmup`` is raw dataset frame ``2 * warmup``.
Getting this wrong leaks the tail of the warmup prefix into post-warmup metrics.
"""

from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.warmup_cutoff import compute_warmup_cutoff_ts


def _write_tum_rgb(dir_path: Path, n: int, t0: float = 100.0, dt: float = 1.0 / 30):
    dir_path.mkdir(parents=True, exist_ok=True)
    lines = ["# color images", "# timestamp filename"]
    lines += [f"{t0 + i * dt:.6f} rgb/{i}.png" for i in range(n)]
    (dir_path / "rgb.txt").write_text("\n".join(lines) + "\n")
    return t0, dt


def _write_euroc_cam0(dir_path: Path, n: int, t0_ns: int = 1000, dt_ns: int = 50_000_000):
    cam0 = dir_path / "mav0" / "cam0"
    cam0.mkdir(parents=True, exist_ok=True)
    lines = ["#timestamp [ns],filename"]
    lines += [f"{t0_ns + i * dt_ns},{t0_ns + i * dt_ns}.png" for i in range(n)]
    (cam0 / "data.csv").write_text("\n".join(lines) + "\n")
    return t0_ns, dt_ns


def _write_kitti_times(path: Path, n: int, dt: float = 0.1):
    path.write_text("\n".join(f"{i * dt:.6e}" for i in range(n)) + "\n")


def test_tum_stride_one_is_raw_index(tmp_path):
    t0, dt = _write_tum_rgb(tmp_path, 100)
    # stride 1: sampled frame 30 == raw frame 30.
    assert compute_warmup_cutoff_ts("tum", tmp_path, 30, stride=1) == pytest.approx(t0 + 30 * dt)


def test_tum_stride_two_uses_doubled_index(tmp_path):
    t0, dt = _write_tum_rgb(tmp_path, 100)
    # stride 2 (DROID/DPVO): sampled frame 30 == raw frame 60. This is the bug
    # the fix addresses -- the old code returned raw frame 30 (1s too early).
    assert compute_warmup_cutoff_ts("tum", tmp_path, 30, stride=2) == pytest.approx(t0 + 60 * dt)
    # and it must differ from the stride-1 answer
    assert compute_warmup_cutoff_ts("tum", tmp_path, 30, stride=2) != pytest.approx(
        compute_warmup_cutoff_ts("tum", tmp_path, 30, stride=1)
    )


def test_euroc_stride_one(tmp_path):
    t0, dt = _write_euroc_cam0(tmp_path, 100)
    # EuRoC cutoff is returned in ns (the trajectory timebase).
    assert compute_warmup_cutoff_ts("euroc", tmp_path, 30, stride=1) == pytest.approx(t0 + 30 * dt)


def test_kitti_stride_one_with_times(tmp_path):
    times = tmp_path / "times.txt"
    _write_kitti_times(times, 200, dt=0.1)
    assert compute_warmup_cutoff_ts(
        "kitti", tmp_path, 30, timestamps_path=times, stride=1
    ) == pytest.approx(30 * 0.1)


def test_warmup_zero_returns_none(tmp_path):
    _write_tum_rgb(tmp_path, 100)
    assert compute_warmup_cutoff_ts("tum", tmp_path, 0, stride=2) is None


def test_index_out_of_range_returns_none(tmp_path):
    # 40 raw frames, stride 2, warmup 30 -> needs raw index 60 -> too short.
    _write_tum_rgb(tmp_path, 40)
    assert compute_warmup_cutoff_ts("tum", tmp_path, 30, stride=2) is None


def test_kitti_without_times_returns_none(tmp_path):
    assert compute_warmup_cutoff_ts("kitti", tmp_path, 30, timestamps_path=None, stride=1) is None


def test_unsupported_dataset_returns_none(tmp_path):
    assert compute_warmup_cutoff_ts("nuscenes", tmp_path, 30, stride=1) is None


def test_metrics_denominator_is_stride_aware():
    """The completeness denominator halves for a stride-2 SLAM: it is the
    sampled post-warmup count (max_frames // stride) - warmup, not the raw
    (max_frames - warmup)."""
    from slamadversariallab.metrics.trajectory import MetricsEvaluator
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        ev1 = MetricsEvaluator(Path(d), "tum", max_frames=500,
                               deadline_warmup_frames=30, deadline_stride=1)
        ev2 = MetricsEvaluator(Path(d), "tum", max_frames=500,
                               deadline_warmup_frames=30, deadline_stride=2)
        # stride 1: 500 - 30 = 470; stride 2: 250 - 30 = 220.
        assert (ev1.max_frames // ev1.deadline_stride) - ev1.deadline_warmup_frames == 470
        assert (ev2.max_frames // ev2.deadline_stride) - ev2.deadline_warmup_frames == 220
