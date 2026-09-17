"""Tests for RuntimeStressOrchestrator.gpu_launch_config()."""

from pathlib import Path

from slamadversariallab.runtime_stress.hami_controller import (
    HAMI_LIB_CONTAINER_PATH,
    HAMI_LIB_HOST_PATH,
)
from slamadversariallab.runtime_stress.models import (
    CpuControl,
    GpuControl,
    MemoryControl,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from slamadversariallab.runtime_stress.orchestrator import RuntimeStressOrchestrator


def _make_orchestrator(request: RuntimeStressRequest, tmp_path: Path) -> RuntimeStressOrchestrator:
    return RuntimeStressOrchestrator(request=request, output_dir=tmp_path)


def test_session_gpu_launch_config_returns_empty_when_no_gpu_controls(tmp_path) -> None:
    request = RuntimeStressRequest(
        scenario_name="cpu_only",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress",
                duration_s=5.0,
                controls=RuntimeStressControls(cpu=CpuControl(max_cores=0.5)),
            )
        ],
        container_runtime="podman",
    )
    orchestrator = _make_orchestrator(request, tmp_path)
    config = orchestrator.gpu_launch_config()
    assert config == {"env": {}, "mounts": [], "devices": []}


def test_session_gpu_launch_config_returns_hami_env_and_mounts_when_gpu_controls_present(
    tmp_path,
) -> None:
    gpu = GpuControl(vram_limit_mb=4096, sm_limit_percent=50)
    request = RuntimeStressRequest(
        scenario_name="gpu_vram_4gb",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress",
                duration_s=5.0,
                controls=RuntimeStressControls(gpu=gpu),
            )
        ],
        container_runtime="podman",
        gpu=gpu,
    )
    orchestrator = _make_orchestrator(request, tmp_path)
    config = orchestrator.gpu_launch_config()

    assert config["env"]["LD_PRELOAD"] == HAMI_LIB_CONTAINER_PATH
    assert config["env"]["CUDA_DEVICE_MEMORY_LIMIT"] == "4096m"
    assert config["env"]["CUDA_DEVICE_SM_LIMIT"] == "50"

    mounts = config["mounts"]
    assert len(mounts) == 1
    src, dst, mode = mounts[0]
    assert src == HAMI_LIB_HOST_PATH
    assert dst == HAMI_LIB_CONTAINER_PATH
    assert mode == "ro"

    assert config["lib_path"] == HAMI_LIB_HOST_PATH
    assert config["devices"] == ["nvidia.com/gpu=all"]


def test_session_gpu_launch_config_with_memory_plus_gpu(tmp_path) -> None:
    gpu = GpuControl(vram_limit_mb=2048)
    request = RuntimeStressRequest(
        scenario_name="gpu_and_memory",
        telemetry_sample_period_ms=500,
        phases=[
            RuntimeStressPhase(
                name="stress",
                duration_s=5.0,
                controls=RuntimeStressControls(
                    memory=MemoryControl(max_mb=512), gpu=gpu
                ),
            )
        ],
        container_runtime="podman",
        gpu=gpu,
    )
    orchestrator = _make_orchestrator(request, tmp_path)
    config = orchestrator.gpu_launch_config()
    assert config["env"]["CUDA_DEVICE_MEMORY_LIMIT"] == "2048m"
    assert "CUDA_DEVICE_SM_LIMIT" not in config["env"]
