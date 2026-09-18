"""S3PO-GS reports its own breakage and the gate must see it.

It prints `Fallback: not enough accurate pixels, apply scale remedy using the
previous keyframe` when it cannot solve scale from the current frame. Validated
across 15 runs, with clean separation and no overlap:

    >= 2 fallbacks -> self-reported ATE 12.31 - 20.93
    <  2 fallbacks -> self-reported ATE  0.48 -  3.97

This was not theoretical. An offline s3pogs cell published 1.022, the mean of
three runs, one of which carried FOUR fallbacks and scored 0.594 -- a broken run
dragging the published mean DOWN and making the system look better than it is.
Over the two clean runs the cell is 1.235.
"""
from __future__ import annotations

import json

from src.runtime_stress.health import verdict_for_run

CLEAN = "current keyframe  21 window is  [21, 16]\nEval: Total FPS 0.9\n"
FALLBACK = "Fallback: not enough accurate pixels, apply scale remedy using the previous keyframe\n"


def _run(tmp_path, log, poses=40):
    (tmp_path / "slam_output.log").write_text(log)
    (tmp_path / "stress_trace.json").write_text(json.dumps([{"phase": "stress"}]))
    (tmp_path / "stress_summary.json").write_text(json.dumps({"exit_code": 0}))
    (tmp_path / "CameraTrajectory.txt").write_text(
        "\n".join("0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0" for _ in range(poses)))
    return tmp_path


def test_two_fallbacks_fails(tmp_path):
    v = verdict_for_run(_run(tmp_path, CLEAN + FALLBACK * 2), "s3pogs")
    assert v["verdict"] == "failed", f"got {v['verdict']}: {v['reasons']}"
    assert any("fallback" in r.lower() for r in v["reasons"])
    assert v["markers"]["scale_fallbacks"] == 2


def test_four_fallbacks_fails(tmp_path):
    """The real case: the run that was published inside a healthy mean."""
    v = verdict_for_run(_run(tmp_path, CLEAN + FALLBACK * 4), "s3pogs")
    assert v["verdict"] == "failed"
    assert v["markers"]["scale_fallbacks"] == 4


def test_one_fallback_is_not_a_failure(tmp_path):
    """The threshold is measured, not chosen. A single fallback occurs in runs
    whose own reported ATE is fine, so failing on one would discard good data."""
    v = verdict_for_run(_run(tmp_path, CLEAN + FALLBACK), "s3pogs")
    assert v["verdict"] != "failed", f"one fallback must not fail: {v['reasons']}"
    assert v["markers"]["scale_fallbacks"] == 1


def test_marker_is_recorded_even_when_clean(tmp_path):
    v = verdict_for_run(_run(tmp_path, CLEAN), "s3pogs")
    assert v["markers"]["scale_fallbacks"] == 0


def test_other_systems_are_unaffected(tmp_path):
    """The branch is s3pogs-only. A fallback line in another system's log must
    not fail it, since the marker has no meaning there."""
    v = verdict_for_run(_run(tmp_path, CLEAN + FALLBACK * 4), "vggtslam")
    assert "scale_fallbacks" not in v["markers"]
