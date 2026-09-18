"""Tests for orchestrator dispatch of the IO controller."""

from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.models import (
    IoControl,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.orchestrator import RuntimeStressOrchestrator
from slamadversariallab.runtime_stress.podman_controllers import PodmanIoController


class _Process:
    def poll(self):
        return None


def _io_request(container_runtime: str = "podman") -> RuntimeStressRequest:
    return RuntimeStressRequest(
        scenario_name="io_only",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress",
                duration_s=1.0,
                controls=RuntimeStressControls(io=IoControl(read_bps=1_000_000)),
            )
        ],
        container_runtime=container_runtime,
    )


def test_orchestrator_dispatches_io_controller_for_podman(tmp_path) -> None:
    orchestrator = RuntimeStressOrchestrator(request=_io_request(), output_dir=tmp_path)
    controllers = orchestrator._build_controllers("podman_container")
    assert any(isinstance(c, PodmanIoController) for c in controllers)


def test_orchestrator_rejects_io_for_docker_target(tmp_path) -> None:
    orchestrator = RuntimeStressOrchestrator(
        request=_io_request(container_runtime="docker"), output_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="podman_container in v1"):
        orchestrator._build_controllers("docker_container")


def test_orchestrator_rejects_io_for_host_process_target(tmp_path) -> None:
    orchestrator = RuntimeStressOrchestrator(request=_io_request(), output_dir=tmp_path)
    with pytest.raises(RuntimeError, match="podman_container in v1"):
        orchestrator._build_controllers("host_process_group")


def test_no_io_controller_when_io_controls_absent(tmp_path) -> None:
    request = RuntimeStressRequest(
        scenario_name="empty",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress", duration_s=1.0, controls=RuntimeStressControls()
            )
        ],
        container_runtime="podman",
    )
    orchestrator = RuntimeStressOrchestrator(request=request, output_dir=tmp_path)
    controllers = orchestrator._build_controllers("podman_container")
    assert all(not isinstance(c, PodmanIoController) for c in controllers)
