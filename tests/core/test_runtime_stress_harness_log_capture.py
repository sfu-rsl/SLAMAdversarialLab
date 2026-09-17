"""The harness must keep a record of what IT did, not only what the SLAM did.

Hard failures were always durable: they raise, `_run_loop` catches them, and
`_record_event("controller_error", ...)` lands them in stress_events.json. Soft
degradations were not. Lines like "initialized_flag stayed 0 ... skipping
mutation for this phase" or "expect read_iops/write_iops caps to silently no-op"
went to a console logger with no file handler attached to any run directory.

That asymmetry is the dangerous one. A run that fails loudly leaves evidence. A
run where the harness quietly applied something other than what the config asked
for left none, so "no warnings found in the logs" was indistinguishable from
"warnings were never written anywhere" -- which is what an audit actually hit.
"""

import json
import logging

from slamadversariallab.runtime_stress.models import (
    CpuControl,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.orchestrator import RuntimeStressOrchestrator

PACKAGE_LOGGER = "slamadversariallab.runtime_stress"


def _request():
    return RuntimeStressRequest(
        scenario_name="capture",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress",
                duration_s=1.0,
                controls=RuntimeStressControls(cpu=CpuControl(max_cores=0.5)),
            )
        ],
    )


def test_controller_warning_is_written_to_harness_log(tmp_path):
    orch = RuntimeStressOrchestrator(request=_request(), output_dir=tmp_path)
    orch._install_log_capture()
    try:
        logging.getLogger(f"{PACKAGE_LOGGER}.hami_controller").warning(
            "initialized_flag stayed 0 after 30.0s; skipping mutation for this phase."
        )
    finally:
        orch._remove_log_capture()

    text = (tmp_path / "harness.log").read_text()
    assert "skipping mutation for this phase" in text
    assert "WARNING" in text


def test_warning_becomes_a_machine_readable_event(tmp_path):
    """A degradation a human has to notice scrolling past is not a control."""
    orch = RuntimeStressOrchestrator(request=_request(), output_dir=tmp_path)
    orch._install_log_capture()
    try:
        logging.getLogger(f"{PACKAGE_LOGGER}.podman_controllers").warning(
            "IOPS throttling on /dev/nvme0n1 may be ineffective with scheduler 'none'."
        )
    finally:
        orch._remove_log_capture()

    degraded = [e for e in orch.events if e.get("kind") == "harness_degraded"]
    assert len(degraded) == 1
    assert degraded[0]["level"] == "WARNING"
    assert "may be ineffective" in degraded[0]["message"]
    assert degraded[0]["logger"].endswith("podman_controllers")


def test_info_is_logged_but_does_not_raise_a_degraded_event(tmp_path):
    """INFO is context, not a degradation. Promoting it would drown the signal
    that verify_treatment is meant to key off."""
    orch = RuntimeStressOrchestrator(request=_request(), output_dir=tmp_path)
    orch._install_log_capture()
    try:
        logging.getLogger(f"{PACKAGE_LOGGER}.hami_controller").info(
            "HAMi runtime mutation: container 'x' limit[0]=4096 bytes (read back OK)"
        )
    finally:
        orch._remove_log_capture()

    assert "read back OK" in (tmp_path / "harness.log").read_text()
    assert [e for e in orch.events if e.get("kind") == "harness_degraded"] == []


def test_capture_is_detached_so_runs_do_not_bleed_into_each_other(tmp_path):
    """A campaign runs many runs in one process. A leaked handler would append
    every later run's output to the first run's file and events list."""
    first, second = tmp_path / "run_0", tmp_path / "run_1"

    a = RuntimeStressOrchestrator(request=_request(), output_dir=first)
    a._install_log_capture()
    a._remove_log_capture()

    b = RuntimeStressOrchestrator(request=_request(), output_dir=second)
    b._install_log_capture()
    try:
        logging.getLogger(f"{PACKAGE_LOGGER}.load_controller").warning("second run only")
    finally:
        b._remove_log_capture()

    assert "second run only" not in (first / "harness.log").read_text()
    assert "second run only" in (second / "harness.log").read_text()
    assert [e for e in a.events if e.get("kind") == "harness_degraded"] == []
    assert len([e for e in b.events if e.get("kind") == "harness_degraded"]) == 1

    # And nothing is left attached to the package logger afterwards.
    remaining = [
        h
        for h in logging.getLogger(PACKAGE_LOGGER).handlers
        if type(h).__name__ == "_HarnessLogCapture"
    ]
    assert remaining == []


def test_degradations_survive_into_the_events_file(tmp_path):
    """The point of the event is that a later check can read it off disk."""
    orch = RuntimeStressOrchestrator(request=_request(), output_dir=tmp_path)
    orch._install_log_capture()
    logging.getLogger(f"{PACKAGE_LOGGER}.hami_controller").warning("cap skipped")
    orch._remove_log_capture()

    # finalize() writes the file; emulate just that step to keep the test unitary.
    (tmp_path / "stress_events.json").write_text(json.dumps(orch.events, indent=2))

    on_disk = json.loads((tmp_path / "stress_events.json").read_text())
    assert any(
        e.get("kind") == "harness_degraded" and "cap skipped" in e.get("message", "")
        for e in on_disk
    )
