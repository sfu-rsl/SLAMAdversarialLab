"""Per-run health verdicts, read from the run's own log.

The rule this enforces (CONTRIBUTING.md, "Run-health verification"): pose
output NEVER counts as success. A SLAM can lose tracking and keep emitting
poses, and a fragmented run's ATE can score BETTER than its own clean baseline,
because each surviving piece is short and internally consistent. So a run's ATE
does not enter any table until that run has a verdict from here.

Until now the gate lived only inside campaign-specific scripts under ``tools/``
(preflight P3), which meant a new campaign silently had no gate at all. This is
the shared implementation those scripts and every future aggregation call.

The catalog behind it is documented in the experiment tracker (P17). Two layers,
because one is not enough:

1. Per-system log markers. Only the ORB-SLAM3 family emits explicit failure
   vocabulary; several systems emit none at all.
2. A fleet-wide derived signal: poses against delivered frames. This must be
   read against the run's own drop count, since under a deadline a low pose
   count reflects dropped frames as much as lost tracking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HEALTHY, DEGRADED, FAILED, UNSCORABLE = "healthy", "degraded", "failed", "unscorable"

# Systems sharing ORB-SLAM3's console vocabulary.
_ORB_FAMILY = {"orbslam3", "orbslam3i", "nitroslam"}

# Any occurrence means the SLAM discarded its map: the run failed, whatever its
# pose count or ATE say.
_RESET_MARKER = "Active map reset recieved"
# S3PO-GS's own breakage marker. Threshold measured, not chosen: see the branch
# in verdict_for_run for the 15-run validation.
_S3POGS = "s3pogs"
_S3POGS_FALLBACK = "Fallback: not enough accurate pixels"
_S3POGS_FALLBACK_LIMIT = 2
# A stumble that recovered. Only counts against the run when NO relocalisation
# follows it.
_TRACK_FAIL = "Fail to track local map!"
_RELOCALIZED = "Relocalized!!"
# Inertial bootstrap completed. Absent in a stressed inertial run = the
# bootstrap never finished, which is the failure mode frame loss actually causes.
_VIBA = "VIBA"

# Fewer than this many post-warmup poses is not a trajectory.
MIN_POSES = 5

# A per-system quantity floor is only meaningful if the system's own clean output
# is large enough to separate collapse from normal. s3pogs produces ~9 clean
# post-warmup poses under a deadline, and its collapsed run produced 9, so no
# floor derived from that anchor can distinguish them. Below this count the gate
# says so instead of pretending a threshold works.
RESOLUTION_FLOOR = 20

# A line each system prints once its own pipeline has finished. Its presence
# means the SLAM's work completed, whatever happened to the process afterwards.
# s3pogs can finish, write its trajectory, print this, and then refuse to exit
# while the container is torn down; a watchdog then force-kills it and the run
# records exit 137 -- the same code a genuine OOM kill produces. Without this
# marker the two are indistinguishable, and a completed run would be filed as a
# crash.
_COMPLETION_MARKERS = {
    "s3pogs": "Total FPS",
    # cuVSLAM's driver prints this last. Needed because the HAMi shim appends
    # its own teardown chatter AFTER the driver finishes, so "last line of the
    # log" is not the completion signal for this system.
    "cuvslam": "cuVSLAM: done",
}
_KILL_EXIT_CODES = {137, -9}

# CUDA-side exhaustion. Distinct from a host OOM kill: one says the GPU memory
# cap bit, the other says the host memory cap did, and the two must be told apart.
_CUDA_OOM_MARKERS = (
    "CUDA out of memory",
    "torch.cuda.OutOfMemoryError",
    "CUDA error: out of memory",
    "cudaErrorMemoryAllocation",
    # Non-torch CUDA runtimes phrase it without the colon, so the existing
    # "CUDA error: out of memory" misses them. cuVSLAM emits
    # "RuntimeError: [CUDA] error out of memory(2)".
    "[CUDA] error out of memory",
)

# DELIBERATELY NOT A CUDA-OOM MARKER: "Device 0 OOM".
#
# That is HAMi's allocator refusing one request, and a SLAM can survive it.
# droidslam's VRAM 75% runs each log it -- asking 7143047552 against a
# 6618611712 cap -- and still finish with 53 and 54 poses against a baseline of
# roughly 55. Treating the line as decisive flips that cell from 50% to 75% and
# makes the system look more fragile than it is, on evidence that is a recovered
# stumble rather than a death. Verified by adding it, watching e2_ordering move,
# and reading the logs.
#
# A HAMi refusal only means failure when it is joined by evidence the run ENDED:
# no trajectory, a truncated pose count, or a non-zero exit. Those channels
# already exist, so this marker would add false positives and no information.


def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return None


def _count_poses(run_dir: Path) -> Optional[int]:
    for name in ("CameraTrajectory.txt", "KeyFrameTrajectory.txt"):
        p = run_dir / name
        text = _read_text(p)
        if text is None:
            continue
        return sum(1 for line in text.splitlines() if line.strip() and not line.startswith("#"))
    return None


def oom_attribution(run_dir: Path, log: Optional[str]) -> Dict[str, Any]:
    """Which memory ran out, if either: the host's or the GPU's.

    Exit codes cannot answer this. 137 and 139 both appear on healthy and on
    destroyed runs, so this would otherwise be inferring "which axis killed it"
    from a signal that does not carry the answer (preflight P6).

    Host side is the kernel's own accounting from cgroup memory.events, taken
    from the run's last telemetry sample. GPU side is the SLAM's own console.
    """
    out: Dict[str, Any] = {"host_oom_kills": None, "cuda_oom": False}

    text = _read_text(run_dir / "stress_trace.json")
    if text:
        try:
            trace = json.loads(text)
            samples = trace if isinstance(trace, list) else (trace.get("samples") or [])
            for sample in reversed(samples):
                ev = sample.get("container_memory_events")
                if isinstance(ev, dict) and ev:
                    out["host_oom_kills"] = ev.get("oom_kill", 0) + ev.get("oom_group_kill", 0)
                    break
        except ValueError:
            pass

    if log:
        out["cuda_oom"] = any(m in log for m in _CUDA_OOM_MARKERS)
    return out


# Uncaught exceptions the SLAM raises about ITSELF. These are death evidence, not
# harness faults, and they must be distinguishable from a watchdog kill.
#
# Keep this list SPECIFIC. A generic "any traceback" rule would catch tracebacks
# printed from except blocks in runs that went on to finish normally, and would
# convert healthy runs into failures.
_FATAL_EXCEPTIONS = {
    "GeometryException":
        "evo could not align the trajectory because it is degenerate, which "
        "means too few or too collinear poses to solve for a transform",
}


def fatal_error_evidence(oom: Dict[str, Any], log: Optional[str]) -> List[str]:
    """Evidence that the run DIED, independent of what its outputs look like.

    Stated as explicit precedence because it decides the s3pogs case by design
    rather than by luck. That run printed a fatal CUDA OOM, then printed its own
    completion marker, then exited 0, and left a 9-pose trajectory. Every
    output-shaped check passed it: 9 clears the pose minimum, its post-warmup
    count sat exactly on the boundary, and its coverage cleared half its clean
    reference because the denominator collapsed alongside the numerator. Only
    the OOM line failed it, so the verdict rested on a console line happening
    to survive.

    Death evidence therefore outranks every output-based check. A run that shows
    it died is failed however healthy its artifacts look.
    """
    found: List[str] = []
    if oom.get("cuda_oom"):
        found.append("CUDA out of memory: the GPU memory cap is what ended this run")
    if oom.get("host_oom_kills"):
        found.append(
            f"{oom['host_oom_kills']} host OOM kill(s): the memory cap is what ended this run"
        )
    # An uncaught exception in the SLAM's own evaluation is death evidence about
    # the SLAM, and it must outrank the harness-kill inference below.
    #
    # WHY THIS IS NEEDED. Some wrappers stop a container that has crashed and
    # then hung (S3PO-GS does: a crash never reaches its completion marker, so
    # its hang workaround cannot fire and the wrapper stops it deliberately).
    # That stop exits on a kill signal with no completion marker, which is
    # exactly the shape the harness-kill branch reads as UNSCORABLE -- "evidence
    # about us, not the SLAM". Without this check, a genuine SLAM crash is filed
    # as missing data. This hit four s3pogs rungs, where the trajectory
    # was so degenerate that evo's alignment raised and the run really had
    # failed.
    if log:
        for marker, why in _FATAL_EXCEPTIONS.items():
            if marker in log:
                found.append(
                    f"{marker}: {why}. The SLAM raised this itself, so the run "
                    f"FAILED rather than being killed by the harness"
                )
    return found


def teardown_kill(run_dir: Path, system: str, log: Optional[str],
                  oom: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Was this run killed AFTER finishing, rather than crashing?

    Returns the evidence when the run printed its own completion marker and
    then exited on a kill signal. That combination is a stuck teardown, and the
    SLAM's results on disk are valid; treating it as a crash would discard good
    data and, worse, would report a harness pathology as a SLAM failure.

    A completion marker is CORROBORATING, never sufficient. s3pogs prints
    ``Total FPS`` after a fatal CUDA OOM and exits 0, so on its own the marker
    proves nothing about whether the work completed. An error line anywhere in
    the log outranks a completion line.
    """
    if oom and fatal_error_evidence(oom, log):
        return None
    marker = _COMPLETION_MARKERS.get(system)
    if not marker or not log or marker not in log:
        return None
    text = _read_text(run_dir / "stress_summary.json")
    if not text:
        return None
    try:
        exit_code = json.loads(text).get("exit_code")
    except ValueError:
        return None
    if exit_code not in _KILL_EXIT_CODES:
        return None
    return {"completed_then_killed": True, "exit_code": exit_code,
            "completion_marker": marker}


def _drop_info(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "deadline_drops.json"
    text = _read_text(p)
    if text is None:
        return {}
    try:
        j = json.loads(text)
    except ValueError:
        return {}
    surv_list = j.get("survivors") or []
    warm = j.get("warmup_frames") or 0
    surv, drop = len(surv_list), len(j.get("dropped") or [])
    total = surv + drop
    return {
        "delivered": surv,
        "dropped": drop,
        "drop_rate": (drop / total) if total else None,
        # Frames delivered AFTER the warmup window. Scoring is post-warmup, so
        # this is a hard upper bound on how many post-warmup poses can exist:
        # a run cannot track frames it was never given.
        "delivered_post_warmup": sum(1 for x in surv_list if x >= warm),
    }


def verdict_for_run(run_dir, system: str,
                    clean_coverage: Optional[float] = None,
                    clean_poses: Optional[int] = None) -> Dict[str, Any]:
    """Judge one run. Returns verdict, reasons, and the markers behind them.

    ``UNSCORABLE`` is distinct from ``FAILED`` on purpose: a failed run is
    evidence about the SLAM, an unscorable one is evidence about our harness
    (no log to read), and the two must never be pooled in a results table.
    """
    run_dir = Path(run_dir)
    system = (system or "").lower()
    reasons: List[str] = []
    markers: Dict[str, Any] = {}

    log = _read_text(run_dir / "slam_output.log")
    if log is None or not log.strip():
        # P18 guarantees a log for every completed run, so its absence means the
        # run did not complete or the wrapper regressed -- not that the SLAM
        # behaved badly.
        #
        # BUT the kernel's own OOM accounting outranks a missing console, because
        # DEATH EVIDENCE OUTRANKS EVERYTHING and a cap that kills a run before it
        # can write anything is the cap working, not a harness fault. okvis2x
        # under a 293 MiB launch-time cap reached ~239 MiB during init and was
        # OOM-killed at t=4.5 s with an empty log and `oom_kill: 1` in the trace;
        # returning "unscorable" there discarded the very result the launch-time
        # protocol exists to produce. Refusing to declare HEALTH without a log is
        # the standing rule; declaring a FAILURE the kernel recorded is not.
        early_oom = oom_attribution(run_dir, log)
        early_death = fatal_error_evidence(early_oom, log)
        if early_death:
            return {
                "verdict": FAILED,
                "reasons": early_death + [
                    "no console output: the cap killed this run before it could log, "
                    "so the kernel's OOM record is the only evidence and it is enough"
                ],
                "markers": {"death_evidence": early_death, "log": "absent",
                            **early_oom},
                "scorable": False,
            }
        return {
            "verdict": UNSCORABLE,
            "reasons": ["no slam_output.log, so the run cannot be health-verified"],
            "markers": {},
            "scorable": False,
        }

    # A run with no telemetry cannot be dose-verified OR stall-classified, so it
    # is not a result either way. This happens when a run is SIGKILLed mid-flight:
    # the trace is written in finalize(), which a killed process never reaches.
    # That is not a bug, but it must be VISIBLE -- an unclassifiable cell that
    # says nothing is the P19 failure shape (a hole that looks like data).
    if not (run_dir / "stress_trace.json").exists():
        return {
            "verdict": UNSCORABLE,
            "reasons": ["no stress_trace.json: the run was killed before it could "
                        "finalise, so it has neither dose verification nor a "
                        "stall classification"],
            "markers": {"telemetry": "absent"},
            "scorable": False,
        }

    # The stressor must be VERIFIED DELIVERED before anything else is read. A
    # cell whose cap never landed is not evidence about the SLAM at all -- it is
    # an unstressed run wearing a stressed label, and its "no effect" reads as a
    # finding. The controller already records this; the gate previously did not
    # look, so invalid cells were being scored as healthy.
    summary_text = _read_text(run_dir / "stress_summary.json")
    if summary_text:
        try:
            ctrl_err = json.loads(summary_text).get("controller_error")
        except ValueError:
            ctrl_err = None
        if ctrl_err:
            return {
                "verdict": UNSCORABLE,
                "reasons": [f"stressor not delivered: {str(ctrl_err)[:160]}"],
                "markers": {"controller_error": str(ctrl_err)[:300]},
                "scorable": False,
            }

    poses = _count_poses(run_dir)
    drops = _drop_info(run_dir)
    oom = oom_attribution(run_dir, log)
    markers.update({"poses": poses, **drops, **oom})

    # Established BEFORE the OOM check, because a post-completion kill also
    # exits 137 and would otherwise read as a host OOM kill.
    text = _read_text(run_dir / "stress_summary.json")
    if text:
        try:
            markers["exit_code"] = json.loads(text).get("exit_code")
        except ValueError:
            pass
    # The marker path is given the death evidence, so a completion line printed
    # AFTER a fatal error cannot be read as proof the work completed.
    killed = teardown_kill(run_dir, system, log, oom)
    if killed:
        markers.update(killed)
        markers["note"] = (
            f"exit {killed['exit_code']} is a post-completion teardown kill, not a "
            f"crash: '{killed['completion_marker']}' was printed and the outputs "
            "were written before the process was force-killed"
        )

    # DEATH EVIDENCE OUTRANKS EVERY OUTPUT-BASED CHECK. Stated precedence, not a
    # coincidence of ordering: a run that shows it died is failed however
    # healthy its artifacts look, and it never reaches the pose, post-warmup or
    # coverage checks that a collapsed-but-tidy run can pass.
    death = fatal_error_evidence(oom, log)
    if death:
        markers["death_evidence"] = death
        return {"verdict": FAILED, "reasons": death, "markers": markers,
                "scorable": False}

    # S3PO-GS reports its own breakage, and until this branch existed the gate
    # could not see it. It prints `Fallback: not enough accurate pixels, apply
    # scale remedy using the previous keyframe` when it cannot solve scale from
    # the current frame, and it also prints its own ATE.
    #
    # VALIDATED ACROSS 15 RUNS, clean separation with no overlap:
    #     >= 2 fallbacks -> self-reported ATE 12.31 - 20.93
    #     <  2 fallbacks -> self-reported ATE  0.48 -  3.97
    #
    # This is not a theoretical gap. An offline s3pogs cell printed 1.022, the
    # mean of three runs, one of which carried FOUR fallbacks and scored 0.594 --
    # a broken run pulling the published mean DOWN and making the system look
    # better than it is. Over the two clean runs the cell is 1.235.
    if system == _S3POGS:
        fallbacks = log.count(_S3POGS_FALLBACK)
        markers["scale_fallbacks"] = fallbacks
        if fallbacks >= _S3POGS_FALLBACK_LIMIT:
            reasons.append(
                f"{fallbacks} scale-remedy fallbacks (limit {_S3POGS_FALLBACK_LIMIT}): "
                f"the SLAM could not solve scale from the current frame and fell "
                f"back to the previous keyframe, which it only does when tracking "
                f"has degraded"
            )

    if system in _ORB_FAMILY:
        resets = log.count(_RESET_MARKER)
        track_fails = log.count(_TRACK_FAIL)
        relocs = log.count(_RELOCALIZED)
        vibas = log.count(_VIBA)
        markers.update({
            "map_resets": resets, "track_fails": track_fails,
            "relocalized": relocs, "viba": vibas,
        })
        if resets:
            reasons.append(f"{resets} map reset(s): the SLAM discarded its map")
        # A stumble only counts when nothing relocalised after it.
        if track_fails and not relocs:
            reasons.append(f"{track_fails} tracking failure(s) with no relocalisation")
        elif track_fails:
            markers["recovered_stumbles"] = min(track_fails, relocs)

    if poses is None:
        # No trajectory file at all. Which of the two this is matters: a run the
        # harness killed before it could finish is evidence about US, while a run
        # that ran to its own end and produced nothing is evidence about the SLAM.
        # Pooling them would let a watchdog kill masquerade as a SLAM failure.
        # A run killed by a cap never reaches here: death evidence returns above,
        # which is what makes the harness-kill inference safe to apply now. A
        # cgroup OOM kill also exits 137 with no completion marker, so without
        # that precedence this branch could not tell the cap's own kill apart
        # from a watchdog, and would file it under "evidence about US".
        marker = _COMPLETION_MARKERS.get(system)
        harness_killed = bool(killed) or (
            marker and marker not in log
            and markers.get("exit_code") in _KILL_EXIT_CODES
        )
        return {
            "verdict": UNSCORABLE if harness_killed else FAILED,
            "reasons": ["no trajectory, and the run was killed before finishing"]
                       if harness_killed else ["no trajectory produced"],
            "markers": markers,
            "scorable": False,
        }
    if poses < MIN_POSES:
        reasons.append(f"{poses} poses is not a trajectory (minimum {MIN_POSES})")

    # Scoring is post-warmup, so a trajectory made almost entirely of warmup
    # poses is not scorable however long the file is. gigaslam writes 26 poses
    # at a 2-core cap of which only 4 fall after the warmup window; counting the
    # whole file called that healthy while the evaluator refused to score it.
    # Two pose counts, two verdicts -- the gate must use the one the rule names.
    # A per-system quantity floor, anchored to what THIS system produces on a
    # clean run under the deadline, and applied only where that anchor has
    # resolution. Below RESOLUTION_FLOOR the anchor cannot separate collapse
    # from normal at all: s3pogs' clean deadline run yields ~9 post-warmup
    # poses and its collapsed VRAM run also yields 9, so any floor drawn from
    # it would pass the collapsed run too. For those systems the honest
    # statement is that health rests on death evidence, recorded here rather
    # than papered over with a threshold that cannot do the work.
    # Measured and rejected: an absolute floor cannot work under a deadline.
    # Anchoring to clean pose count was tried and re-scored against 144
    # runs: it moved 13 verdicts healthy -> failed and every one was FALSE.
    # okvis2x at half a core produced 119 poses from 120 delivered frames, and
    # at one core 232 from 233 -- it tracked every frame it was given. dpvslam
    # managed 55 from 55. Their pose counts fell because the DEADLINE delivered
    # fewer frames, which is the treatment working, not the SLAM failing.
    # Absolute counts are therefore not comparable across severities, and only
    # a delivered-normalised ratio is -- which is exactly the coverage check
    # below. So the quantity check IS coverage, and this records the reference
    # without judging on it.
    if clean_poses:
        markers["clean_poses_reference"] = clean_poses
        markers["poses_vs_clean"] = round(poses / clean_poses, 2)
    if not clean_poses or clean_poses < RESOLUTION_FLOOR:
        markers["quantity_check"] = (
            "no resolution: this system's clean-deadline pose count is too low for any "
            "quantity comparison to distinguish collapse from normal, so health rests "
            "on death evidence and log markers"
        )

    pw = drops.get("delivered_post_warmup")
    if pw is not None and drops.get("delivered") and pw < MIN_POSES:
        reasons.append(
            f"only {pw} frames delivered after warmup (minimum {MIN_POSES}); the "
            "trajectory is a warmup-only fragment and cannot be scored post-warmup"
        )

    if reasons:
        return {"verdict": FAILED, "reasons": reasons, "markers": markers, "scorable": False}

    # No failure marker. A thin trajectory is only evidence of damage RELATIVE to
    # what this system produces when healthy: the keyframe- and submap-based
    # systems emit far fewer poses than delivered frames by construction, so an
    # absolute threshold measures their architecture, not their health. VGGT
    # tracks ~12% of delivered frames on a clean run and ~10-13% at every
    # severity, and calling that degradation flagged 42 of 120 runs wrongly.
    # Without a reference, coverage is recorded and NOT judged.
    if poses is not None and drops.get("delivered"):
        coverage = poses / drops["delivered"]
        markers["coverage_of_delivered"] = round(coverage, 3)
        if clean_coverage:
            markers["coverage_vs_clean"] = round(coverage / clean_coverage, 2)
        # Half of its own clean coverage is a real collapse; a fixed fraction of
        # delivered frames is not.
        if clean_coverage and coverage < 0.5 * clean_coverage:
            return {
                "verdict": DEGRADED,
                "reasons": [
                    f"tracked {coverage:.0%} of delivered frames against a clean "
                    f"reference of {clean_coverage:.0%} for this system, with no "
                    "failure marker in the log; read by hand before scoring"
                ],
                "markers": markers,
                "scorable": True,
            }

    return {"verdict": HEALTHY, "reasons": [], "markers": markers, "scorable": True}


def gate_ate(verdict: Dict[str, Any], ate: Optional[float]) -> Optional[float]:
    """Return the ATE only when the verdict permits scoring, else None.

    Callers should print ``None`` as a cross, never as a blank that reads like
    missing data: a failed run HAS an ATE, and that number is exactly the one
    that misleads, since it is computed across disconnected map pieces with no
    common reference frame.
    """
    return ate if verdict.get("scorable") else None


# A cell may be called stall-bound only on BOTH pieces of evidence together.
STALL_TAIL_S = 120.0      # look at the last two minutes of telemetry
STALL_MIN_SAMPLES = 8


def stall_binding(run_dir) -> Dict[str, Any]:
    """Did a capped run STOP MAKING PROGRESS, or was it merely slow?

    A timed-out run is stall-bound only when the cgroup shows it pinned at its
    limit and reclaiming (memory.events ``max``/``high``) AND frames stopped
    advancing over the tail of the run. Either alone is not enough:

    - Reclaim without stagnation is a system paying a price and still working.
    - Stagnation without reclaim is a hang that has nothing to do with the cap,
      and attributing it to the axis under test would invent a finding.

    Without this conjunction a short timeout silently converts "slow" into
    "bound", which is the same silent-verdict class as the coverage threshold
    and the ambiguous exit code.
    """
    # "Bound" always needs the conjunction. "NOT bound" may rest on reclaim=False
    # alone -- but only for a run that reached its own end, because reclaim is a
    # cumulative counter and a completed run with zero reclaim never approached
    # its limit. A KILLED run with an unreadable tail tells us nothing: we did
    # not see its end, so absence of reclaim so far is not evidence of absence.
    # Before trace flushing this case could not arise (a killed run had no trace
    # at all); now that partial traces survive, it can.
    finalized = (Path(run_dir) / "stress_summary.json").exists()
    out = {"stall_bound": False, "reclaim": None, "frames_advanced_in_tail": None,
           "determinate": finalized}
    text = _read_text(Path(run_dir) / "stress_trace.json")
    if not text:
        return out
    try:
        trace = json.loads(text)
    except ValueError:
        return out
    samples = trace if isinstance(trace, list) else (trace.get("samples") or [])
    if not samples:
        return out

    last_ev = next((x["container_memory_events"] for x in reversed(samples)
                    if isinstance(x.get("container_memory_events"), dict)
                    and x["container_memory_events"]), None)
    reclaim = bool(last_ev and (last_ev.get("max", 0) or last_ev.get("high", 0)))
    out["reclaim"] = reclaim

    end_t = max((x.get("elapsed_s") or 0) for x in samples)
    tail = [x for x in samples
            if (x.get("elapsed_s") or 0) >= end_t - STALL_TAIL_S
            and x.get("deadline_frame") is not None]
    if len(tail) < STALL_MIN_SAMPLES:
        # No readable frame tail. Whether that is judgeable depends on RECLAIM,
        # and conflating the two cases overstates what we know:
        #
        #   reclaim FALSE on a finalized run -> determinate "not bound", because
        #       a run that reached its end without ever approaching its limit
        #       cannot have been pinned at it.
        #   reclaim TRUE with no frame tail  -> INDETERMINATE. The conjunction's
        #       first term is satisfied and its second is unmeasured, so "slow"
        #       and "bound" are indistinguishable. Reporting False here reads as
        #       "not bound" and silently turns missing evidence into a negative.
        #
        # This is what orbslam3i's 25% launch-time runs hit: reclaim=True, no
        # frame tail, previously reported stall_bound=False with determinate=True,
        # which would have made a bracket look non-reproducing when the truth is
        # that it cannot be judged from these runs.
        if reclaim:
            out["determinate"] = False
        return out

    advanced = tail[-1]["deadline_frame"] - tail[0]["deadline_frame"]
    out["frames_advanced_in_tail"] = advanced
    out["stall_bound"] = bool(reclaim and advanced == 0)
    # A readable tail makes the judgement determinate even on a killed run.
    out["determinate"] = True
    return out
