"""Contract tests for SLAMAlgorithm.run() fail-fast behavior."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from slamadverseriallab.algorithms.base import ExecutionSpec, SLAMAlgorithm
from slamadverseriallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRunResult,
    SLAMRuntimeContext,
    SensorMode,
)


class _DummyAlgorithm(SLAMAlgorithm):
    """Minimal algorithm used to validate base run() behavior."""

    def __init__(self, stage_error: Optional[Exception] = None):
        self.stage_error = stage_error
        self.calls: List[str] = []

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

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        self.calls.append("preflight")

    def _stage_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        self.calls.append("stage")
        if self.stage_error is not None:
            raise self.stage_error
        return request.dataset_path

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        self.calls.append("cleanup_stage")

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, Any]]:
        self.calls.append("build_execution_inputs")
        return {"dataset_path": request.dataset_path}

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        return None

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        self.calls.append("execute")
        return False

    def _find_raw_trajectory(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Any]:
        self.calls.append("find_raw_trajectory")
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Any,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        self.calls.append("convert_trajectory")
        return None


def _build_request(tmp_path: Path) -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="seq_01",
        extras={},
    )


def test_run_fails_fast_on_stage_exception_with_exact_message(tmp_path: Path) -> None:
    algo = _DummyAlgorithm(stage_error=RuntimeError("stage failed hard"))
    result: SLAMRunResult = algo.run(_build_request(tmp_path))

    assert result.success is False
    assert result.trajectory_path is None
    assert result.message == "stage failed hard"


def test_run_stage_exception_skips_execute_and_trajectory_collection(tmp_path: Path) -> None:
    algo = _DummyAlgorithm(stage_error=RuntimeError("stage exploded"))
    _ = algo.run(_build_request(tmp_path))

    assert "build_execution_inputs" not in algo.calls
    assert "execute" not in algo.calls
    assert "find_raw_trajectory" not in algo.calls
    assert "convert_trajectory" not in algo.calls


def test_run_stage_exception_still_runs_cleanup(tmp_path: Path) -> None:
    algo = _DummyAlgorithm(stage_error=RuntimeError("stage broken"))
    _ = algo.run(_build_request(tmp_path))

    assert "cleanup_stage" in algo.calls
