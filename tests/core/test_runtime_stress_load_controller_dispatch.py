"""Orchestrator dispatch + finalize wiring for the load-antagonist controller.

House pattern from test_runtime_stress_io_controller_dispatch.py: build a
real RuntimeStressOrchestrator from a hand-built request and assert on
_build_controllers / finalize behavior.
"""

import json
import tempfile
from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.load_controller import PodmanLoadController
from slamadversariallab.runtime_stress.models import (
    CpuControl,
    LoadControl,
    LoadFence,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.podman_controllers import PodmanCpuController
from slamadversariallab.runtime_stress.orchestrator import RuntimeStressOrchestrator


def _request(with_load: bool) -> RuntimeStressRequest:
    controls = RuntimeStressControls(
        load=LoadControl(cpu_workers=8, fence=LoadFence(cpus=4.0)) if with_load else None
    )
    return RuntimeStressRequest(
        scenario_name="load_dispatch",
        telemetry_sample_period_ms=250,
        phases=[
            RuntimeStressPhase(name="warmup", duration_s=1.0),
            RuntimeStressPhase(name="stress", duration_s=10.0, controls=controls),
        ],
        container_runtime="podman",
    )


def test_dispatches_load_controller_for_podman():
    orch = RuntimeStressOrchestrator(_request(True), Path(tempfile.mkdtemp()))
    controllers = orch._build_controllers("podman_container")
    load_controllers = [c for c in controllers if isinstance(c, PodmanLoadController)]
    assert len(load_controllers) == 1
    assert orch._load_controller is load_controllers[0]
    # Appended last: slow GPU-init prepare must not delay cap attachment.
    assert isinstance(controllers[-1], PodmanLoadController)


def test_absent_without_load_controls():
    orch = RuntimeStressOrchestrator(_request(False), Path(tempfile.mkdtemp()))
    controllers = orch._build_controllers("podman_container")
    assert not any(isinstance(c, PodmanLoadController) for c in controllers)
    assert orch._load_controller is None


@pytest.mark.parametrize("kind", ["docker_container", "host_process_group"])
def test_unsupported_targets_fail_loud(kind):
    orch = RuntimeStressOrchestrator(_request(True), Path(tempfile.mkdtemp()))
    with pytest.raises(RuntimeError, match="podman_container"):
        orch._build_controllers(kind)


def test_cap_and_load_compose_in_one_phase():
    """A phase may declare a cap on the SLAM AND a load antagonist (the
    realistic quota-plus-noisy-neighbor deployment): both controllers must
    dispatch, caps first, load last."""
    controls = RuntimeStressControls(
        cpu=CpuControl(max_cores=2.0),
        load=LoadControl(cpu_workers=8, fence=LoadFence(cpus=4.0)),
    )
    req = RuntimeStressRequest(
        scenario_name="cap_plus_load",
        telemetry_sample_period_ms=250,
        phases=[RuntimeStressPhase(name="stress", duration_s=10.0, controls=controls)],
        container_runtime="podman",
    )
    orch = RuntimeStressOrchestrator(req, Path(tempfile.mkdtemp()))
    controllers = orch._build_controllers("podman_container")
    kinds = [type(c).__name__ for c in controllers]
    assert "PodmanCpuController" in kinds and "PodmanLoadController" in kinds
    assert kinds.index("PodmanCpuController") < kinds.index("PodmanLoadController")


def test_prewarm_constructs_controller_and_build_reuses_it(monkeypatch):
    orch = RuntimeStressOrchestrator(_request(True), Path(tempfile.mkdtemp()))
    monkeypatch.setattr(PodmanLoadController, "prewarm_gpu", lambda self: False)
    orch.prewarm_load_antagonists()
    prewarmed = orch._load_controller
    assert prewarmed is not None
    controllers = orch._build_controllers("podman_container")
    assert orch._load_controller is prewarmed
    assert any(c is prewarmed for c in controllers)


def test_prewarm_noop_without_load_controls():
    orch = RuntimeStressOrchestrator(_request(False), Path(tempfile.mkdtemp()))
    orch.prewarm_load_antagonists()
    assert orch._load_controller is None


def test_prewarm_failure_sets_controller_error(monkeypatch):
    orch = RuntimeStressOrchestrator(_request(True), Path(tempfile.mkdtemp()))

    def boom(self):
        raise RuntimeError("antagonist image missing")

    monkeypatch.setattr(PodmanLoadController, "prewarm_gpu", boom)
    monkeypatch.setattr(PodmanLoadController, "cleanup", lambda self: None)
    with pytest.raises(RuntimeError, match="image missing"):
        orch.prewarm_load_antagonists()
    assert "prewarm failed" in orch._controller_error
    assert any(e.get("kind") == "controller_error" for e in orch.events)


def test_finalize_snapshots_load_antagonists():
    out = Path(tempfile.mkdtemp())
    orch = RuntimeStressOrchestrator(_request(True), out)

    class _Stub(PodmanLoadController):
        def antagonist_summaries(self):
            return [{"kind": "stress_ng", "container_name": "sal-load-sng-x"}]

    orch._load_controller = _Stub()
    orch.finalize(execution_ok=True, trajectory_found=True)
    summary = json.loads((out / "stress_summary.json").read_text())
    assert summary["load_antagonists"] == [
        {"kind": "stress_ng", "container_name": "sal-load-sng-x"}
    ]
