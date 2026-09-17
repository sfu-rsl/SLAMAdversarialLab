"""The health gate: what each verdict is allowed to mean.

This gate decides which runs are permitted to contribute a number, so a wrong
verdict does not produce a wrong measurement — it produces a confident finding
about something that never happened. That has already occurred once on this
campaign: four runs whose stressor never landed scored healthy, and a false
result ("okvis2x's memory axis does not bind") was read off them and had to be
retracted. Until now the gate had no tests at all.

The distinction these tests exist to protect:

    failed      evidence about the SLAM. It broke under the dose.
    unscorable  evidence about the HARNESS. The dose never landed, or we
                killed the run before it could answer.

Pooling them corrupts the result in both directions. Counting harness kills as
failures invents fragility that is ours, not the system's. Counting the cap's
own kills as harness noise quietly subtracts successes from the axis being
measured.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.runtime_stress.health import verdict_for_run


# Benign console content. Non-empty on purpose: the gate refuses to
# health-verify a run with no log at all, and a fixture that left it empty
# would make tests pass on that rule instead of the one under test.
BENIGN_LOG = "loading config\nrunning\n"


def _run(tmp, *, log=BENIGN_LOG, exit_code=0, poses=None, memory_events=None,
         controller_error=None, trajectory=True):
    """Build a run directory shaped like a real one."""
    d = Path(tmp)
    (d / "slam_output.log").write_text(log)

    summary = {"exit_code": exit_code, "trajectory_found": bool(poses)}
    if controller_error:
        summary["controller_error"] = controller_error
    (d / "stress_summary.json").write_text(json.dumps(summary))

    sample = {"elapsed_s": 1.0, "phase": "stress"}
    if memory_events is not None:
        sample["container_memory_events"] = memory_events
    (d / "stress_trace.json").write_text(json.dumps([sample]))

    if poses is not None:
        (d / "CameraTrajectory.txt").write_text(
            "\n".join(f"{i} 0 0 0 0 0 0 1" for i in range(poses)))
    return d


OOM = {"high": 0, "max": 0, "oom": 0, "oom_kill": 1, "oom_group_kill": 0}
NO_OOM = {"high": 0, "max": 28, "oom": 0, "oom_kill": 0, "oom_group_kill": 0}


# ---------------------------------------------------------------------------
# The dose has to land before anything else is read
# ---------------------------------------------------------------------------

def test_a_run_whose_stressor_never_landed_is_unscorable():
    # The retraction case. This run looks entirely normal from the outside, and
    # was previously scored healthy -- an unstressed run wearing a stressed
    # label, which reads as "no effect" and is worse than no data.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, poses=500, exit_code=0,
                 controller_error="Command '['podman', 'update', '--memory', "
                                  "'293m', 'c']' timed out after 10 seconds")
        v = verdict_for_run(d, "okvis2x")
    assert v["verdict"] == "unscorable"
    assert not v["scorable"]
    assert any("not delivered" in r for r in v["reasons"])


def test_dose_failure_outranks_a_healthy_looking_trajectory():
    # Ordering matters: the dose check must come before pose counting, or a
    # full trajectory would carry the run past the gate.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, poses=5000, exit_code=0, controller_error="timed out after 10 seconds")
        v = verdict_for_run(d, "orbslam3i")
    assert v["verdict"] == "unscorable"


# ---------------------------------------------------------------------------
# Whose kill was it
# ---------------------------------------------------------------------------

def test_an_oom_kill_is_the_cap_working_not_harness_noise():
    # exit 137 with no trajectory and no completion marker is also what a
    # watchdog kill looks like. The kernel's own OOM accounting is what tells
    # them apart, and it has to win -- otherwise the memory cap's successes are
    # filed as our failures and the axis looks weaker than it is.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="Killed\n", exit_code=137, poses=None, memory_events=OOM)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] == "failed"
    assert any("memory cap" in r for r in v["reasons"])


def test_a_kill_with_no_oom_evidence_stays_unscorable():
    # Same exit code, same missing trajectory, but nothing attributes the death
    # to the cap. Guessing here is what the gate exists to prevent.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, exit_code=137, poses=None, memory_events=NO_OOM)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] == "unscorable"


def test_cuda_oom_is_attributed_to_the_gpu_not_the_host():
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="torch.cuda.OutOfMemoryError: CUDA out of memory\n",
                 exit_code=137, poses=None, memory_events=NO_OOM)
        v = verdict_for_run(d, "droidslam")
    assert v["verdict"] == "failed"
    assert any("GPU memory cap" in r for r in v["reasons"])


def test_a_completed_run_killed_during_teardown_is_not_a_crash():
    # s3pogs finishes, writes its trajectory, prints its marker, then refuses
    # to exit and is force-killed. Exit 137 here means nothing about the SLAM.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="Total FPS: 12.4\n", exit_code=137, poses=400)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] != "failed"


# ---------------------------------------------------------------------------
# Pose output is never on its own a pass
# ---------------------------------------------------------------------------

def test_a_run_that_produced_nothing_and_was_not_killed_is_a_failure():
    # Ran to its own end, produced no trajectory. That is the SLAM's answer.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="done\n", exit_code=0, poses=None)
        v = verdict_for_run(d, "droidslam")
    assert v["verdict"] == "failed"
    assert not v["scorable"]


def test_a_handful_of_poses_is_not_a_trajectory():
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, exit_code=0, poses=2)
        v = verdict_for_run(d, "droidslam")
    assert v["verdict"] != "healthy"


def test_an_orb_map_reset_fails_the_run_however_many_poses_it_wrote():
    # The standing rule, and the reason pose count cannot be the gate: a
    # fragmented run can out-score its own clean baseline on ATE.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="Active map reset recieved\n", exit_code=0, poses=3000)
        v = verdict_for_run(d, "orbslam3i")
    assert v["verdict"] == "failed"


# ---------------------------------------------------------------------------
# Stated precedence: death evidence outranks every output-based check
# ---------------------------------------------------------------------------

def test_a_tidy_looking_run_that_shows_it_died_is_failed():
    """The s3pogs VRAM case, which used to be caught only by luck.

    That run printed a fatal CUDA OOM, then printed its own completion marker,
    then exited 0, leaving a 9-pose trajectory. Every output-shaped check passed
    it. Only the OOM line failed it, so the verdict rested on a console line
    happening to survive. Death evidence is now stated precedence.
    """
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="torch.cuda.OutOfMemoryError: CUDA out of memory\n"
                        "Eval: Total FPS 0.61\n",
                 exit_code=0, poses=9)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] == "failed"
    assert not v["scorable"]


def test_a_completion_marker_after_a_fatal_error_proves_nothing():
    # The marker is corroborating, never sufficient: an error line anywhere
    # outranks a completion line, so the teardown path must not claim this run
    # completed its work.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="torch.cuda.OutOfMemoryError: CUDA out of memory\n"
                        "Eval: Total FPS 0.61\n",
                 exit_code=137, poses=400)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] == "failed"
    assert "completed_then_killed" not in (v.get("markers") or {})


def test_a_clean_teardown_kill_is_still_not_a_failure():
    # Same marker, same kill code, but NO error evidence. This one really did
    # finish, and must not be swept up by the rule above.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, log="Eval: Total FPS 12.4\n", exit_code=137, poses=400)
        v = verdict_for_run(d, "s3pogs")
    assert v["verdict"] != "failed"
    assert (v.get("markers") or {}).get("completed_then_killed")


# ---------------------------------------------------------------------------
# The quantity floor, and its stated resolution limit
# ---------------------------------------------------------------------------

def test_an_absolute_pose_count_never_fails_a_run_on_its_own():
    """Measured and rejected: absolute counts cannot work under a deadline.

    Anchoring to clean pose count was tried and re-scored against E1's 144 runs.
    It moved 13 verdicts to failed and every one was FALSE: okvis2x produced 119
    poses from 120 delivered frames at half a core, and 232 from 233 at one core,
    having tracked every frame it was given. Their counts fell because the
    deadline delivered fewer frames, which is the treatment working. So the
    reference is recorded and never judged on.
    """
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, exit_code=0, poses=30)
        v = verdict_for_run(d, "droidslam", clean_poses=100)
    assert v["verdict"] == "healthy"
    assert (v.get("markers") or {}).get("poses_vs_clean") == 0.3


def test_the_floor_is_declined_when_the_anchor_lacks_resolution():
    # s3pogs' clean deadline run yields ~9 post-warmup poses and its COLLAPSED
    # run also yielded 9. A floor drawn from that anchor would pass the
    # collapsed run, so the gate must say the check cannot work here rather
    # than pretend a threshold does the job.
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, exit_code=0, poses=9)
        v = verdict_for_run(d, "s3pogs", clean_poses=9)
    assert "no resolution" in (v.get("markers") or {}).get("quantity_check", "")


def test_no_anchor_supplied_also_declines_rather_than_inventing_one():
    with tempfile.TemporaryDirectory() as t:
        d = _run(t, exit_code=0, poses=9)
        v = verdict_for_run(d, "droidslam")
    assert "no resolution" in (v.get("markers") or {}).get("quantity_check", "")
