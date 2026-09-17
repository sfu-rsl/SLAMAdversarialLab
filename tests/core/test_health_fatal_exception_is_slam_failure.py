"""A SLAM that raises must be FAILED, not UNSCORABLE.

The gate separates two things on purpose: FAILED is evidence about the SLAM,
UNSCORABLE is evidence about our harness (a watchdog kill, a missing log). They
must never be pooled, because an unscorable cell is a hole in the data while a
failed one is a result.

Some wrappers stop a container that has crashed and then hung. S3PO-GS is one: a
crash never reaches the completion marker its hang workaround watches for, so the
wrapper stops it deliberately. That stop exits on a kill signal with no
completion marker -- exactly the shape the harness-kill branch reads as
UNSCORABLE. Without an explicit check, a genuine crash is filed as missing data.

E4NW hit this on four s3pogs rungs: the trajectory was so degenerate that evo's
alignment raised, and the run really had failed.
"""
from __future__ import annotations

import json

import pytest

from src.runtime_stress.health import fatal_error_evidence, verdict_for_run

TRACEBACK = (
    "current keyframe  26 window is  [26, 0]\n"
    "Traceback (most recent call last):\n"
    '  File "/usr/local/lib/python3.10/dist-packages/evo/core/geometry.py", line 35\n'
    "    raise GeometryException(\"Degenerate covariance rank, \")\n"
    "evo.core.geometry.GeometryException: Degenerate covariance rank, "
    "Umeyama alignment is not possible\n"
)


def test_fatal_exception_is_death_evidence():
    found = fatal_error_evidence({}, TRACEBACK)
    assert found, "an uncaught GeometryException must count as death evidence"
    assert "GeometryException" in found[0]
    assert "FAILED rather than being killed by the harness" in found[0]


def test_clean_log_yields_no_death_evidence():
    assert fatal_error_evidence({}, "current keyframe 21\nEval: Total FPS 0.19\n") == []


def test_unrelated_traceback_is_not_death_evidence():
    """Only the named exceptions count. A generic traceback rule would convert
    healthy runs -- ones that print a traceback from an except block and then
    finish -- into failures."""
    log = ("Traceback (most recent call last):\n"
           '  File "/opt/s3pogs/retry.py", line 12\n'
           "ValueError: transient parse failure, retrying\n"
           "Eval: Total FPS 0.19\n")
    assert fatal_error_evidence({}, log) == []


def _run(tmp_path, log, exit_code=137):
    (tmp_path / "slam_output.log").write_text(log)
    (tmp_path / "stress_trace.json").write_text(json.dumps([{"phase": "stress"}]))
    (tmp_path / "stress_summary.json").write_text(json.dumps({"exit_code": exit_code}))
    return tmp_path


def test_crashed_run_scores_failed_not_unscorable(tmp_path):
    """The whole point: a killed-looking crash must still read as a SLAM failure.

    No trajectory file, exit 137, no completion marker -- the exact shape of a
    harness kill. The exception in the log is what separates them.
    """
    v = verdict_for_run(_run(tmp_path, TRACEBACK), "s3pogs")
    assert v["verdict"] == "failed", f"expected failed, got {v['verdict']}: {v['reasons']}"
    assert any("GeometryException" in r for r in v["reasons"])


def test_harness_kill_without_an_exception_stays_unscorable(tmp_path):
    """The distinction must survive: a kill with no SLAM-side error is still
    evidence about US, and must not be laundered into a SLAM failure."""
    v = verdict_for_run(_run(tmp_path, "current keyframe 21\nstill working\n"), "s3pogs")
    assert v["verdict"] != "failed", (
        f"a plain harness kill must not be reported as a SLAM failure: {v}")
