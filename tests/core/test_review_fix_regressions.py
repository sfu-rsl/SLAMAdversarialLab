"""Regression tests for the post-campaign review fixes.

Locks in three behaviors that, if reverted, silently corrupt results:

1. Stale deadline/progress logs from a previous run into the same output dir
   are cleared for realtime AND non-realtime runs (`_realtime_env`); a stale
   drop log would otherwise remap DROID poses through a dead run's survivor
   list.
2. DROID's `_staged_rgb` staging dir is rebuilt from scratch each run; stale
   hardlinks would otherwise pin old image content or extra frames.
3. A frame-anchored orchestrator run that never sees the SLAM's progress file
   is marked invalid (controller_error) after a grace window instead of
   silently publishing phase-0-only results as fully stressed.
"""

import json
import tempfile
from pathlib import Path

from slamadversariallab.pipelines.runtime_stress_evaluation import (
    PROGRESS_FILENAME,
    _realtime_env,
)
from slamadversariallab.runtime_stress.models import (
    RealtimeDeadline,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.orchestrator import (
    FRAME_PROGRESS_GRACE_S,
    RuntimeStressOrchestrator,
)


# ---------------------------------------------------------------------------
# 1. _realtime_env stale-log clearing
# ---------------------------------------------------------------------------

def _seed_stale_logs(tmp_path: Path) -> tuple[Path, Path]:
    drop_log = tmp_path / "deadline_drops.json"
    progress = tmp_path / PROGRESS_FILENAME
    drop_log.write_text(json.dumps({"survivors": [0, 2], "dropped": [1], "total_items": 3}))
    progress.write_text(json.dumps({"frame": 999}))
    return drop_log, progress


def test_realtime_env_clears_stale_logs_for_non_realtime_run(tmp_path):
    drop_log, progress = _seed_stale_logs(tmp_path)
    with _realtime_env(None, drop_log):
        assert not drop_log.exists(), "stale drop log must be cleared for non-realtime runs"
        assert not progress.exists(), "stale progress file must be cleared for non-realtime runs"


def test_realtime_env_clears_stale_logs_for_realtime_run(tmp_path):
    drop_log, progress = _seed_stale_logs(tmp_path)
    realtime = RealtimeDeadline(target_fps=20.0)
    with _realtime_env(realtime, drop_log):
        assert not drop_log.exists()
        assert not progress.exists()


# ---------------------------------------------------------------------------
# 2. DROID _staged_rgb rebuild
# ---------------------------------------------------------------------------

def _stage(algo, output_dir: Path, paths):
    return algo._stage_approved_frames(output_dir, [str(p) for p in paths])


def test_staged_rgb_drops_frames_no_longer_approved(tmp_path):
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    algo = DROIDSLAMAlgorithm()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    a, b = src_dir / "a.png", src_dir / "b.png"
    a.write_bytes(b"A1")
    b.write_bytes(b"B1")

    out = tmp_path / "run"
    staging = _stage(algo, out, [a, b])
    assert sorted(p.name for p in staging.iterdir()) == ["a.png", "b.png"]

    # Second run approves only a.png: b.png must NOT linger in staging.
    staging = _stage(algo, out, [a])
    assert sorted(p.name for p in staging.iterdir()) == ["a.png"]


def test_staged_rgb_picks_up_regenerated_content(tmp_path):
    from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm

    algo = DROIDSLAMAlgorithm()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    img = src_dir / "frame.png"
    img.write_bytes(b"OLD")

    out = tmp_path / "run"
    staging = _stage(algo, out, [img])
    assert (staging / "frame.png").read_bytes() == b"OLD"

    # Regenerate the source (same filename, new inode/content), as a
    # re-run of the perturbation pipeline does.
    img.unlink()
    img.write_bytes(b"NEW")
    staging = _stage(algo, out, [img])
    assert (staging / "frame.png").read_bytes() == b"NEW", (
        "staging must re-link regenerated content, not keep stale hardlinks"
    )


# ---------------------------------------------------------------------------
# 3. Frame-anchored progress-stall guard
# ---------------------------------------------------------------------------

def _frame_request() -> RuntimeStressRequest:
    return RuntimeStressRequest(
        scenario_name="frame_anchored",
        telemetry_sample_period_ms=250,
        phases=[
            RuntimeStressPhase(name="a", until_frame=100, controls=RuntimeStressControls()),
            RuntimeStressPhase(name="b", until_frame=200, controls=RuntimeStressControls()),
        ],
        container_runtime="podman",
        realtime=RealtimeDeadline(target_fps=15.0),
    )


def _orch() -> RuntimeStressOrchestrator:
    return RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))


def test_stall_guard_quiet_within_grace():
    orch = _orch()
    orch._check_progress_stall(FRAME_PROGRESS_GRACE_S - 1)
    assert orch._controller_error is None


def test_stall_guard_fails_loud_after_grace():
    orch = _orch()
    orch._check_progress_stall(FRAME_PROGRESS_GRACE_S + 1)
    assert orch._controller_error is not None
    assert "invalid" in orch._controller_error
    error_events = [e for e in orch.events if e.get("kind") == "error"]
    assert len(error_events) == 1

    # Reported once, not per sample.
    orch._check_progress_stall(FRAME_PROGRESS_GRACE_S + 2)
    assert len([e for e in orch.events if e.get("kind") == "error"]) == 1


def test_stall_guard_quiet_once_progress_seen(tmp_path):
    orch = _orch()
    progress = tmp_path / "deadline_progress.json"
    progress.write_text(json.dumps({"frame": 3}))
    orch._progress_path = str(progress)
    assert orch._read_progress_frame() == 3
    assert orch._progress_seen is True

    orch._check_progress_stall(FRAME_PROGRESS_GRACE_S + 100)
    assert orch._controller_error is None


def test_stall_guard_ignores_time_anchored_runs():
    req = RuntimeStressRequest(
        scenario_name="time",
        telemetry_sample_period_ms=250,
        phases=[RuntimeStressPhase(name="s", duration_s=10.0)],
    )
    orch = RuntimeStressOrchestrator(req, Path(tempfile.mkdtemp()))
    orch._check_progress_stall(FRAME_PROGRESS_GRACE_S + 100)
    assert orch._controller_error is None
