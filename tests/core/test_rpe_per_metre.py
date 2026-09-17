"""RPE per metre: the companion that survives frame drops.

The reported RPE counts its delta in ENTRIES OF THE TRAJECTORY FILE. When drops
thin the file, five entries span more real motion, so the same delta measures a
longer baseline and accumulates more error -- geometry, not tracking quality.

Measured on okvis2x's E1 ladder, where ATE says quality is unchanged:

    poses   delta=5 spans   RPE(frames)   RPE(1 m)
      499         0.25 s        0.0081      0.0807
      200         0.63 s        0.0214      0.0803
       97         1.26 s        0.0510      0.0847
                                 6.3x        1.1x

These tests pin the three properties that made it trustworthy enough to wire in.
"""
from __future__ import annotations

import numpy as np
import pytest

from evo.core import metrics as evo_metrics
from evo.core.trajectory import PoseTrajectory3D

from src.metrics.trajectory import _rpe_delta_metres, _rpe_per_metre


def _straight_line(n, spacing, noise=0.0, seed=0):
    """A trajectory travelling +x, optionally with per-pose noise."""
    rng = np.random.default_rng(seed)
    xyz = np.zeros((n, 3))
    xyz[:, 0] = np.arange(n) * spacing
    if noise:
        xyz += rng.normal(0.0, noise, xyz.shape)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return PoseTrajectory3D(positions_xyz=xyz, orientations_quat_wxyz=quat,
                            timestamps=np.arange(n, dtype=float) * 0.05)


def test_delta_scales_with_the_dataset():
    """A desk sequence and a street sequence cannot share one metre delta."""
    assert _rpe_delta_metres("tum_freiburg1_desk") == 0.1
    assert _rpe_delta_metres("euroc_V1_01_easy") == 1.0
    assert _rpe_delta_metres("kitti_07") == 5.0
    # Unknown names must not raise; a safe default beats a crash in a companion.
    assert _rpe_delta_metres("something_new") == 1.0
    assert _rpe_delta_metres("") == 1.0


def test_pose_density_does_not_change_the_result():
    """THE PROPERTY THIS METRIC EXISTS FOR.

    Two trajectories over the same route and the same ground truth, one sampled
    half as densely -- exactly what a frame drop produces. A frames-delta RPE
    moves because five entries span twice as far. A metres-delta RPE must not.
    """
    ref = _straight_line(200, 0.05)
    dense = _straight_line(200, 0.05, noise=0.002, seed=1)
    sparse_ref = PoseTrajectory3D(
        positions_xyz=ref.positions_xyz[::2],
        orientations_quat_wxyz=ref.orientations_quat_wxyz[::2],
        timestamps=ref.timestamps[::2])
    sparse = PoseTrajectory3D(
        positions_xyz=dense.positions_xyz[::2],
        orientations_quat_wxyz=dense.orientations_quat_wxyz[::2],
        timestamps=dense.timestamps[::2])

    a = _rpe_per_metre(ref, dense, "euroc")["rpe_m_rmse"]
    b = _rpe_per_metre(sparse_ref, sparse, "euroc")["rpe_m_rmse"]
    assert a is not None and b is not None
    # Halving the density must not move it much. The frames-delta equivalent
    # roughly doubles here, which is the whole reason this exists.
    assert b == pytest.approx(a, rel=0.5), (
        f"metres-RPE moved from {a} to {b} when only pose DENSITY changed")


def test_it_refuses_a_trajectory_too_short_to_measure():
    """A collapsed run must not be scored.

    This is the opposite of ATE's behaviour, which returned 0.0672 for an
    orbslam3i cell that had failed with map resets. A refusal is more honest
    than a number computed over a route the system never travelled.
    """
    tiny = _straight_line(3, 0.001)          # 3 mm of travel, delta is 1 m
    out = _rpe_per_metre(tiny, tiny, "euroc")
    assert out["rpe_m_rmse"] is None
    assert out.get("rpe_m_error"), "a refusal must say why"
    assert out["rpe_m_delta"] == 1.0, "the attempted delta is still recorded"


def test_a_companion_failure_never_raises():
    """It must not take down the reported RPE it accompanies.

    Mismatched lengths rather than an empty trajectory: evo raises when
    CONSTRUCTING an empty one, so that never reaches this function. Length
    mismatch is a failure it genuinely has to survive, and it is the shape a
    synchronisation bug upstream would produce.
    """
    ref = _straight_line(60, 0.05)
    est = _straight_line(30, 0.05)
    out = _rpe_per_metre(ref, est, "euroc")   # must not raise
    assert out["rpe_m_rmse"] is None
    assert out.get("rpe_m_error"), "a failure must say why"
