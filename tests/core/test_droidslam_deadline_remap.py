"""Verify DROID-SLAM trajectory conversion correctly remaps timestamps
when the SAL deadline harness was active during the run.

DROID-SLAM stores its sequential frame counter in ``data["tstamps"]``;
that counter indexes into whatever stream the SLAM actually iterated.
When the deadline harness skipped frames, DROID's counter no longer
matches the position in the original sampled stream, so naively using
it as an index into the timestamp list would yield wrong timestamps.
The fix in ``_convert_reconstruction`` reads the survivor list from
``deadline_drops.json`` and remaps the indices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest


@pytest.fixture
def fake_reconstruction(tmp_path):
    """Save a torch reconstruction.pth with known tstamps and poses."""
    import torch

    def _make(tstamps: List[int]):
        n = len(tstamps)
        # Identity quaternion (qx=qy=qz=0, qw=1) for every pose; positions
        # are just sequential so we can verify ordering survives.
        poses = np.zeros((n, 7), dtype=np.float32)
        poses[:, 0] = np.arange(n)         # tx
        poses[:, 6] = 1.0                  # qw
        data = {
            "tstamps": torch.tensor(tstamps, dtype=torch.float32),
            "poses": torch.tensor(poses, dtype=torch.float32),
        }
        recon_path = tmp_path / "reconstruction.pth"
        torch.save(data, str(recon_path))
        return recon_path

    return _make


def _stride_two_timestamps(num_dataset_frames: int) -> Dict[int, float]:
    """timestamps_by_frame for TUM-like data at 30 Hz with stride=2.

    Frame i has timestamp i / 30.0 seconds. The DROID-SLAM TUM stride
    is 2 (set in droidslam.py:_resolve_stride for dataset_type='tum').
    """
    return {i: i / 30.0 for i in range(num_dataset_frames)}


def _read_traj_timestamps(traj_path: Path) -> List[float]:
    out = []
    for line in traj_path.read_text().strip().split("\n"):
        out.append(float(line.split()[0]))
    return out


def test_no_drop_log_uses_tstamps_directly(fake_reconstruction, tmp_path):
    """Without a drop log, conversion must behave exactly as before
    (DROID's t == position in the strided stream)."""
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    # 20 dataset frames, stride=2, so sampled stream is positions 0,2,4,...,18 (10 frames).
    timestamps_by_frame = _stride_two_timestamps(20)
    # DROID processed all 10 sampled frames; tstamps = [0,1,2,...,9].
    recon_path = fake_reconstruction(list(range(10)))

    algo = DROIDSLAMAlgorithm.__new__(DROIDSLAMAlgorithm)
    ok = algo._convert_reconstruction(
        recon_path, tmp_path, "tum", stride=2,
        timestamps_by_frame=timestamps_by_frame,
    )
    assert ok

    traj_ts = _read_traj_timestamps(tmp_path / "CameraTrajectory.txt")
    # Sampled timestamps at stride 2 are 0/30, 2/30, 4/30, ..., 18/30.
    expected = [i * 2 / 30.0 for i in range(10)]
    assert traj_ts == pytest.approx(expected)


def test_drop_log_remaps_via_survivors(fake_reconstruction, tmp_path):
    """With a drop log present, DROID's tstamps must be looked up
    through the survivor list before mapping to dataset timestamps."""
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    timestamps_by_frame = _stride_two_timestamps(20)

    # SAL deadline harness skipped sampled positions 1, 3, 4 in this run.
    # Survivor list (positions DROID actually iterated, in order) is:
    survivors = [0, 2, 5, 6, 7, 8, 9]
    (tmp_path / "deadline_drops.json").write_text(json.dumps({
        "survivors": survivors,
        "dropped": [1, 3, 4],
        "total_items": 10,
        "target_fps": 10.0,
    }))

    # DROID processed 7 frames (matching len(survivors)); tstamps are 0..6.
    recon_path = fake_reconstruction(list(range(len(survivors))))

    algo = DROIDSLAMAlgorithm.__new__(DROIDSLAMAlgorithm)
    ok = algo._convert_reconstruction(
        recon_path, tmp_path, "tum", stride=2,
        timestamps_by_frame=timestamps_by_frame,
    )
    assert ok

    traj_ts = _read_traj_timestamps(tmp_path / "CameraTrajectory.txt")
    # For each DROID t in 0..6, look up survivors[t] -> position in
    # sampled stream, then convert to dataset timestamp:
    # survivors -> sampled position -> dataset frame -> timestamp
    # 0 -> 0 -> 0  -> 0/30
    # 1 -> 2 -> 4  -> 4/30
    # 2 -> 5 -> 10 -> 10/30
    # 3 -> 6 -> 12 -> 12/30
    # 4 -> 7 -> 14 -> 14/30
    # 5 -> 8 -> 16 -> 16/30
    # 6 -> 9 -> 18 -> 18/30
    expected = [s * 2 / 30.0 for s in survivors]
    assert traj_ts == pytest.approx(expected)


def test_malformed_drop_log_fails_conversion(fake_reconstruction, tmp_path):
    """A present-but-corrupt drop log must fail the conversion loud (return
    False + logged error/traceback), NOT silently identity-remap. DROID keys
    poses by survivor counter, so an identity map over a real deadline run
    would misalign every pose against ground truth."""
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    timestamps_by_frame = _stride_two_timestamps(20)
    (tmp_path / "deadline_drops.json").write_text("not valid json {{{")
    recon_path = fake_reconstruction(list(range(10)))

    algo = DROIDSLAMAlgorithm.__new__(DROIDSLAMAlgorithm)
    ok = algo._convert_reconstruction(
        recon_path, tmp_path, "tum", stride=2,
        timestamps_by_frame=timestamps_by_frame,
    )
    assert not ok
    assert not (tmp_path / "CameraTrajectory.txt").exists()


def test_empty_survivors_fails_conversion(fake_reconstruction, tmp_path):
    """A drop log with empty survivors is a truncated/corrupt log (a real log
    always records at least the warmup survivors), so it must fail loud rather
    than silently identity-remap."""
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    timestamps_by_frame = _stride_two_timestamps(20)
    (tmp_path / "deadline_drops.json").write_text(json.dumps({
        "survivors": [],
        "dropped": [],
        "total_items": 0,
    }))
    recon_path = fake_reconstruction(list(range(10)))

    algo = DROIDSLAMAlgorithm.__new__(DROIDSLAMAlgorithm)
    ok = algo._convert_reconstruction(
        recon_path, tmp_path, "tum", stride=2,
        timestamps_by_frame=timestamps_by_frame,
    )
    assert not ok
