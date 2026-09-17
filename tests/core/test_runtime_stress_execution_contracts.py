"""Contract tests for runtime-stress execution handling."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from slamadversariallab.algorithms.base import ExecutionSpec, SLAMAlgorithm
from slamadversariallab.algorithms.types import SLAMRuntimeContext, SensorMode


class _DummyAlgorithm(SLAMAlgorithm):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {"tum": ["mono"]}

    def cleanup(self) -> None:
        return

    def resolve_config_name(
        self,
        sequence: str,
        dataset_type: str,
        sensor_mode: Optional[SensorMode] = None,
    ) -> Optional[str]:
        return "dummy"

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return None

    def _preflight_checks(self, request, ctx) -> None:
        return

    def _build_execution_inputs(self, request, ctx) -> Optional[Dict[str, Any]]:
        return {}

    def _build_execution_spec(self, request, ctx) -> Optional[ExecutionSpec]:
        return None

    def _execute(self, request, ctx) -> bool:
        return False

    def _find_raw_trajectory(self, request, ctx):
        return None

    def _convert_raw_trajectory_to_tum(self, raw_trajectory, request, ctx):
        return None


def test_run_execution_spec_rejects_unknown_target_when_runtime_stress_active() -> None:
    algo = _DummyAlgorithm()
    algo._active_runtime_context = type(
        "_Ctx",
        (),
        {"runtime_stress_session": object()},
    )()

    spec = ExecutionSpec(
        cmd=["echo", "hello"],
        custom_runner=lambda _spec: True,
        target_kind="custom_target",
    )

    with pytest.raises(RuntimeError, match="does not support execution target kind"):
        algo._run_execution_spec(spec)


def test_run_execution_spec_attaches_podman_target_metadata_when_runtime_stress_active(
    monkeypatch,
) -> None:
    algo = _DummyAlgorithm()

    attached = {}

    class _Session:
        def attach_process(self, process, target_kind: str, target_metadata=None) -> None:
            attached["process"] = process
            attached["target_kind"] = target_kind
            attached["target_metadata"] = target_metadata

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 1234
            self.returncode = 0

        def wait(self, timeout=None) -> None:
            self.returncode = 0

        def poll(self):
            return self.returncode

    fake_process = _FakeProcess()
    monkeypatch.setattr(
        "slamadversariallab.algorithms.base.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    algo._active_runtime_context = type(
        "_Ctx",
        (),
        {"runtime_stress_session": _Session()},
    )()

    spec = ExecutionSpec(
        cmd=["podman", "run", "orbslam3:latest"],
        stream_output=False,
        target_kind="podman_container",
        target_metadata={"container_name": "orbslam3-test"},
    )

    returncode = algo._run_execution_spec(spec)

    assert returncode == 0
    assert attached["process"] is fake_process
    assert attached["target_kind"] == "podman_container"
    assert attached["target_metadata"] == {"container_name": "orbslam3-test"}


def test_run_execution_spec_attaches_docker_target_metadata_when_runtime_stress_active(
    monkeypatch,
) -> None:
    algo = _DummyAlgorithm()

    attached = {}

    class _Session:
        def attach_process(self, process, target_kind: str, target_metadata=None) -> None:
            attached["process"] = process
            attached["target_kind"] = target_kind
            attached["target_metadata"] = target_metadata

    class _FakeProcess:
        def __init__(self) -> None:
            self.pid = 1234
            self.returncode = 0

        def wait(self, timeout=None) -> None:
            self.returncode = 0

        def poll(self):
            return self.returncode

    fake_process = _FakeProcess()
    monkeypatch.setattr(
        "slamadversariallab.algorithms.base.subprocess.Popen",
        lambda *args, **kwargs: fake_process,
    )

    algo._active_runtime_context = type(
        "_Ctx",
        (),
        {"runtime_stress_session": _Session()},
    )()

    spec = ExecutionSpec(
        cmd=["docker", "run", "orbslam3:latest"],
        stream_output=False,
        target_kind="docker_container",
        target_metadata={"container_name": "orbslam3-test"},
    )

    returncode = algo._run_execution_spec(spec)

    assert returncode == 0
    assert attached["process"] is fake_process
    assert attached["target_kind"] == "docker_container"
    assert attached["target_metadata"] == {"container_name": "orbslam3-test"}
