"""Tests for frame-anchored runtime-stress phases.

A frame-anchored phase switches controls when the SLAM reaches a target
sampled-frame index (reported live by the DeadlineIterator) rather than at
a wall-clock second. This keeps a control change pinned to the same point
in the trajectory regardless of the paced frame rate.
"""

import json
import tempfile
from pathlib import Path

from slamadversariallab.runtime_stress.models import (
    CpuControl,
    RealtimeDeadline,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.orchestrator import RuntimeStressOrchestrator


def _frame_request() -> RuntimeStressRequest:
    return RuntimeStressRequest(
        scenario_name="frame_anchored",
        telemetry_sample_period_ms=250,
        phases=[
            RuntimeStressPhase(name="a", until_frame=100, controls=RuntimeStressControls()),
            RuntimeStressPhase(name="b", until_frame=200, controls=RuntimeStressControls()),
            RuntimeStressPhase(name="c", until_frame=300, controls=RuntimeStressControls()),
        ],
        container_runtime="podman",
        realtime=RealtimeDeadline(target_fps=15.0),
    )


def test_request_frame_anchored_flag():
    assert _frame_request().frame_anchored is True


def test_request_time_anchored_flag():
    req = RuntimeStressRequest(
        scenario_name="time",
        telemetry_sample_period_ms=250,
        phases=[RuntimeStressPhase(name="s", duration_s=10.0)],
    )
    assert req.frame_anchored is False


def test_phase_index_for_frame_selects_by_boundary():
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    # phase a: [0,100), b: [100,200), c: [200,300), clamp to c past 300.
    assert orch._phase_index_for_frame(0) == 0
    assert orch._phase_index_for_frame(99) == 0
    assert orch._phase_index_for_frame(100) == 1
    assert orch._phase_index_for_frame(199) == 1
    assert orch._phase_index_for_frame(200) == 2
    assert orch._phase_index_for_frame(299) == 2
    assert orch._phase_index_for_frame(5000) == 2


def test_read_progress_frame_reads_file(tmp_path):
    progress = tmp_path / "deadline_progress.json"
    progress.write_text(json.dumps({"frame": 142}))
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._progress_path = str(progress)
    assert orch._read_progress_frame() == 142


def test_read_progress_frame_never_rewinds(tmp_path):
    progress = tmp_path / "deadline_progress.json"
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._progress_path = str(progress)

    progress.write_text(json.dumps({"frame": 50}))
    assert orch._read_progress_frame() == 50

    # A stale/older read must not move phase selection backwards.
    progress.write_text(json.dumps({"frame": 30}))
    assert orch._read_progress_frame() == 50


def test_read_progress_frame_tolerates_missing_and_corrupt(tmp_path):
    progress = tmp_path / "deadline_progress.json"
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._progress_path = str(progress)

    # Missing file -> last known (0).
    assert orch._read_progress_frame() == 0

    # Half-written / corrupt JSON -> last known, no raise.
    progress.write_text('{"frame": 7')  # truncated
    assert orch._read_progress_frame() == 0


def test_read_progress_frame_no_path_returns_last():
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._progress_path = None
    assert orch._read_progress_frame() == 0


def test_apply_phase_stamps_triggering_frame_into_event():
    """A frame-anchored phase transition records the frame that triggered it."""
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._controllers = []  # no real controllers in a unit test
    orch._apply_phase(1, elapsed_s=4.2, frame=100)
    enter = [e for e in orch.events if e["kind"] == "phase_enter"]
    assert enter and enter[-1]["message"] == "b"
    assert enter[-1]["frame"] == 100


def test_apply_phase_omits_frame_for_time_anchored():
    """Time-anchored transitions don't carry a frame field."""
    orch = RuntimeStressOrchestrator(_frame_request(), Path(tempfile.mkdtemp()))
    orch._controllers = []
    orch._apply_phase(0, elapsed_s=1.0)  # frame defaults to None
    enter = [e for e in orch.events if e["kind"] == "phase_enter"]
    assert enter and "frame" not in enter[-1]


def test_realtime_env_clears_stale_progress_and_drop_logs(tmp_path):
    """Entering the realtime context deletes leftover logs from a prior run.

    Regression: a stale progress file from a previous run that reused the
    output dir made the orchestrator read a high frame count immediately
    and skip straight to the last phase.
    """
    from slamadversariallab.pipelines.runtime_stress_evaluation import (
        _realtime_env,
        PROGRESS_FILENAME,
    )
    from slamadversariallab.runtime_stress.models import RealtimeDeadline

    drop_log = tmp_path / "deadline_drops.json"
    progress = tmp_path / PROGRESS_FILENAME
    drop_log.write_text('{"survivors": [0], "dropped": []}')
    progress.write_text('{"frame": 99}')  # stale, from a "previous run"

    realtime = RealtimeDeadline(target_fps=15.0)
    with _realtime_env(realtime, drop_log):
        assert not drop_log.exists()
        assert not progress.exists()
