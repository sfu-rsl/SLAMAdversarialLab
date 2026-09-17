"""A metric must record the inputs that produced it, not only the answer.

An ape.json holding six statistics cannot be audited. A run aligned at scale
0.247, whose trajectory was 4x too small and got resized to fit, wrote a file
indistinguishable from one aligned at 1.00 while reporting an error that read as
healthy. A run scored over 4 associated poses wrote the same shape of file as one
scored over 1385. The alignment scale was computed, logged at debug, and thrown
away.

The same shape bit three other ways in one day: two tools rounded one mean
differently and both numbers reached a PDF, two quantities both called "poses"
diverged by up to 22, and a broken run was averaged into a published ATE. Every
one was invisible because the artifact recorded the answer and not the inputs.
"""

import json
from pathlib import Path

import pytest

from slamadversariallab.metrics.trajectory import MetricsEvaluator

GT = Path("datasets/euroc/V1_01_easy/mav0/state_groundtruth_estimate0/data.csv")
RUN = Path(
    "results/e1/camp_c0base_orbslam3i/slam_results/orbslam3i/runtime_stress"
    "/nocap/run_0/CameraTrajectory.txt"
)

# These assert on a REAL recorded run and its ground truth, neither of which is
# distributed: `results/` is generated, and `datasets/` is fetched separately. In
# a fresh clone the whole module skips rather than failing, because a reader
# following CONTRIBUTING's "run the tests" step should not be told the install is
# broken when it is only missing data they were never given.
pytestmark = pytest.mark.skipif(
    not (RUN.exists() and GT.exists()),
    reason=f"needs a recorded run ({RUN}) and its ground truth ({GT}); "
           "run the campaign first, or fetch the dataset",
)

STATS = {"rmse", "mean", "median", "std", "min", "max"}
PROVENANCE = {
    "n_poses",
    "n_poses_estimated",
    "association_coverage",
    "scale",
    "correct_scale",
}


def _ape(tmp_path):
    ev = MetricsEvaluator(tmp_path, "euroc")
    return ev._compute_ape(RUN, GT, "baseline_run_0")


def _rpe(tmp_path):
    ev = MetricsEvaluator(tmp_path, "euroc")
    return ev._compute_rpe(RUN, GT, "baseline_run_0")


def test_ape_records_the_inputs_beside_the_answer(tmp_path):
    m = _ape(tmp_path)
    assert STATS <= set(m), "the statistics must survive unchanged"
    assert PROVENANCE <= set(m), f"missing provenance: {PROVENANCE - set(m)}"


def test_the_scale_factor_is_recorded(tmp_path):
    """The one number that says whether the error can be trusted."""
    m = _ape(tmp_path)
    assert isinstance(m["scale"], float)
    # A healthy stereo-inertial run sits near 1. The point is not the value but
    # that it is now recoverable at all: 0.247 and 1.00 must not write the same
    # file.
    assert 0.5 < m["scale"] < 2.0


def test_pose_count_and_coverage_are_recorded(tmp_path):
    m = _ape(tmp_path)
    assert m["n_poses"] > 0
    assert m["n_poses"] <= m["n_poses_estimated"]
    assert 0.0 < m["association_coverage"] <= 1.0


def test_the_alignment_convention_is_recorded_not_inferred(tmp_path):
    """Sim(3) versus SE(3) is a methodological choice. A future reader should
    not have to infer which one produced a number from the code as it stands
    then."""
    assert _ape(tmp_path)["correct_scale"] is True


def test_provenance_reaches_disk(tmp_path):
    """In-memory is not the point. The stored artifact is what gets audited."""
    _ape(tmp_path)
    written = json.loads((tmp_path / "run_0" / "baseline" / "ape.json").read_text())
    assert PROVENANCE <= set(written)
    assert written["n_poses"] > 0


def test_rpe_records_the_delta_it_actually_used(tmp_path):
    """RPE falls back from delta=5 to delta=1 on a sparse trajectory, which
    changes the quantity rather than the precision. Before this was recorded the
    fallback showed up only as a console warning, so two cells measured over
    different intervals could be compared as though they were not."""
    m = _rpe(tmp_path)
    assert m["delta"] in (1, 5)
    assert m["delta_unit"] == "frames"
    written = json.loads((tmp_path / "run_0" / "baseline" / "rpe.json").read_text())
    assert written["delta"] == m["delta"]
    assert PROVENANCE <= set(written)


# --- The SE(3) companion -----------------------------------------------------
#
# The reported metric aligns with Sim(3), which resizes the estimate before
# measuring. Correct for monocular, where scale is unobservable. Wrong for a
# stereo or inertial system, which can observe scale, where the resize absorbs
# real error. Rather than change the reported number, the honest one is stored
# beside it and the gap between them is the signal.


def test_se3_companion_is_recorded(tmp_path):
    m = _ape(tmp_path)
    assert "rmse_se3" in m and "se3_error" in m
    assert m["se3_error"] is None, "a healthy run should align under SE(3)"
    assert isinstance(m["rmse_se3"], float)


def test_se3_is_never_better_than_the_reported_value(tmp_path):
    """The resize is a free parameter fitted to minimise error, so it can only
    equal or beat the un-resized fit. If this ever inverts, the alignment is not
    doing what the code believes."""
    m = _ape(tmp_path)
    assert m["rmse_se3"] >= m["rmse"] - 1e-9


def test_a_healthy_run_shows_no_gap(tmp_path):
    """Scale near 1 means the resize had nothing to absorb, so the two agree."""
    m = _ape(tmp_path)
    assert abs(m["scale"] - 1.0) < 0.01
    assert m["rmse_se3"] / m["rmse"] < 1.02


def test_the_gap_exposes_a_hidden_scale_error(tmp_path):
    """The case this exists for. A 3-pose run whose trajectory is 12 percent
    oversized reports the best error in its campaign under Sim(3). The SE(3)
    companion is 2.3x larger, which is what says the number cannot be used."""
    run = Path(
        "results/e1/camp_c2dlcpu_orbslam3i/slam_results/orbslam3i/runtime_stress"
        "/dlcpu_H_1core/run_0/CameraTrajectory.txt"
    )
    if not run.exists():
        import pytest
        pytest.skip("fixture run not present")
    m = MetricsEvaluator(tmp_path, "euroc")._compute_ape(run, GT, "baseline_run_0")
    assert m["scale"] > 1.1
    assert m["rmse_se3"] > 2 * m["rmse"]


def test_rpe_carries_the_companion_at_the_same_delta(tmp_path):
    """Comparing an RPE at delta=5 against a companion at delta=1 would compare
    two different quantities, which is the H11 defect in a new place."""
    m = _rpe(tmp_path)
    assert "rmse_se3" in m
    written = json.loads((tmp_path / "run_0" / "baseline" / "rpe.json").read_text())
    assert written["delta"] == m["delta"]
    assert "rmse_se3" in written


def test_a_failed_companion_records_why_and_spares_the_metric(tmp_path):
    """An SE(3) failure must not take the reported metric down with it, and the
    reason must be recorded rather than swallowed."""
    import slamadversariallab.metrics.trajectory as T

    def boom():
        raise ValueError("degenerate")

    out = T._se3_companion(object(), object(), boom, "x")
    assert out["rmse_se3"] is None
    assert out["se3_error"] in ("ValueError", "TypeError", "AttributeError")
