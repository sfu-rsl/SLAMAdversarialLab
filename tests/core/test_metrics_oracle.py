"""Numerical oracle + metamorphic property tests for the ATE/RPE metrics stack.

These validate the metrics against closed-form answers and mathematical
invariants, end to end through MetricsEvaluator (association + Sim3 alignment +
APE/RPE), rather than tracing one code path. A regression in alignment,
association, or the error computation breaks a law here even if no single
example looks wrong.
"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

np_random = np.random.RandomState(7)
Rot = pytest.importorskip("scipy.spatial.transform").Rotation

from slamadversariallab.metrics.trajectory import MetricsEvaluator  # noqa: E402


def _tum(ts, pos, quat_xyzw):
    return "\n".join(
        f"{ts[i]:.6f} {pos[i,0]:.6f} {pos[i,1]:.6f} {pos[i,2]:.6f} "
        f"{quat_xyzw[i,0]:.6f} {quat_xyzw[i,1]:.6f} {quat_xyzw[i,2]:.6f} {quat_xyzw[i,3]:.6f}"
        for i in range(len(ts))
    )


def _make_gt(n=200, seed=1):
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 20, n)
    pos = np.c_[np.sin(t), np.cos(0.7 * t), 0.1 * t] + rng.normal(0, 0.01, (n, 3))
    # (N, 1), not (N,). scipy 1.18 tightened from_euler: a single-axis sequence
    # with a flat array of N angles is rejected, where earlier versions broadcast
    # it. The column form is accepted by both, so this stays portable across the
    # `scipy>=1.9` range requirements.txt allows.
    quat = Rot.from_euler("z", (0.3 * t)[:, None]).as_quat()
    return t, pos, quat


def _ate(tmp, ts_e, pos_e, q_e, ts_g, pos_g, q_g):
    tmp = Path(tmp)
    (tmp / "gt.txt").write_text(_tum(ts_g, pos_g, q_g))
    (tmp / "est.txt").write_text(_tum(ts_e, pos_e, q_e))
    ev = MetricsEvaluator(tmp / "out", "tum")
    return ev._compute_ape(tmp / "est.txt", tmp / "gt.txt", "prop_run_0")["rmse"]


def test_oracle_recovers_injected_noise():
    """Rigid transform + known Gaussian noise -> ATE recovers the noise RMS."""
    t, gp, gq = _make_gt()
    n = len(t)
    R = Rot.from_euler("xyz", [0.5, -0.3, 1.1]); tv = np.array([3.0, -2.0, 1.5])
    sigma = 0.05
    ep = R.apply(gp) + tv + np_random.normal(0, sigma, gp.shape)
    eq = (R * Rot.from_quat(gq)).as_quat()
    expected = np.sqrt(3) * sigma * np.sqrt((n - 7) / n)  # 7 alignment DOF
    with tempfile.TemporaryDirectory() as d:
        got = _ate(d, t, ep, eq, t, gp, gq)
    assert abs(got - expected) < 0.01, f"got {got}, expected ~{expected}"


def test_zero_error_is_zero():
    """est == GT -> ATE == 0."""
    t, gp, gq = _make_gt()
    with tempfile.TemporaryDirectory() as d:
        assert _ate(d, t, gp, gq, t, gp, gq) < 1e-6


def test_pure_rigid_transform_aligns_to_zero():
    """A noise-free rigid transform of GT must align back to ~0 ATE."""
    t, gp, gq = _make_gt()
    R = Rot.from_euler("xyz", [1.2, 0.4, -0.9]); tv = np.array([-5.0, 4.0, 2.0])
    ep = R.apply(gp) + tv
    eq = (R * Rot.from_quat(gq)).as_quat()
    with tempfile.TemporaryDirectory() as d:
        assert _ate(d, t, ep, eq, t, gp, gq) < 1e-4


def test_sim3_absorbs_global_scale():
    """Under scale-corrected (Sim3) alignment, a globally scaled GT aligns to ~0."""
    t, gp, gq = _make_gt()
    ep = 2.5 * gp  # global scale, no rotation/translation/noise
    with tempfile.TemporaryDirectory() as d:
        assert _ate(d, t, ep, gq, t, gp, gq) < 1e-4


def test_noise_is_monotonic_in_ate():
    """More injected noise -> strictly larger ATE."""
    t, gp, gq = _make_gt()
    vals = []
    for sigma in (0.02, 0.05, 0.10, 0.20):
        ep = gp + np.random.RandomState(int(sigma * 1000)).normal(0, sigma, gp.shape)
        with tempfile.TemporaryDirectory() as d:
            vals.append(_ate(d, t, ep, gq, t, gp, gq))
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1)), vals


def test_rigid_invariance_of_ate():
    """ATE is invariant to applying the SAME rigid transform to est and GT."""
    t, gp, gq = _make_gt()
    ep = gp + np.random.RandomState(3).normal(0, 0.05, gp.shape)
    R = Rot.from_euler("xyz", [0.7, -1.1, 0.2]); tv = np.array([10.0, -3.0, 5.0])
    gp2 = R.apply(gp) + tv; gq2 = (R * Rot.from_quat(gq)).as_quat()
    ep2 = R.apply(ep) + tv; eq2 = (R * Rot.from_quat(gq)).as_quat()
    with tempfile.TemporaryDirectory() as d:
        a1 = _ate(d, t, ep, gq, t, gp, gq)
    with tempfile.TemporaryDirectory() as d:
        a2 = _ate(d, t, ep2, eq2, t, gp2, gq2)
    assert abs(a1 - a2) < 1e-4, f"{a1} vs {a2}"
