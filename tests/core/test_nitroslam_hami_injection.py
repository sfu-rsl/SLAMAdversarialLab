"""Tests for HAMi env/mount/device injection into Nitro-SLAM container commands.

Unlike ORB-SLAM3 (CPU-only, where HAMi env is inert plumbing), Nitro-SLAM is a
real CUDA workload, so the HAMi env/mounts/devices are injected live and the GPU
is always attached.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from slamadversariallab.algorithms.nitroslam import NitroSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)
from slamadversariallab.runtime_stress.hami_controller import HAMI_SHARED_CACHE


class _FakeSession:
    def __init__(self, env: Dict[str, str], mounts, devices=None):
        self._env = env
        self._mounts = mounts
        self._devices = list(devices or [])

    def gpu_launch_config(self) -> Dict[str, Any]:
        return {
            "env": dict(self._env),
            "mounts": list(self._mounts),
            "devices": list(self._devices),
        }


def _build_request(tmp_path: Path) -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="EuRoC.yaml",
        output_dir=output_dir,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
    )


def _build_context(
    request: SLAMRunRequest,
    staged_path: Path,
    image_mounts: Optional[List[Tuple[str, str]]] = None,
) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="EuRoC.yaml",
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_path,
    )
    ctx.execution_inputs = {
        "dataset_path": staged_path,
        "output_dir": request.output_dir,
        "euroc_image_mounts": list(image_mounts or []),
    }
    ctx.runtime_stress = object()
    return ctx


def _build_spec_with_session(algo, ctx, session):
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = session
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def _make(tmp_path: Path):
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = NitroSLAMAlgorithm(container_runtime="podman")
    return algo, ctx


def test_nitroslam_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, ctx = _make(tmp_path)
    session = _FakeSession(
        env={
            "LD_PRELOAD": "/opt/hami/libvgpu.so",
            "CUDA_DEVICE_MEMORY_LIMIT": "6144m",
            "CUDA_DEVICE_SM_LIMIT": "100",
        },
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "-e" in spec.cmd
    assert "LD_PRELOAD=/opt/hami/libvgpu.so" in spec.cmd
    assert "CUDA_DEVICE_MEMORY_LIMIT=6144m" in spec.cmd
    assert "CUDA_DEVICE_SM_LIMIT=100" in spec.cmd


def test_nitroslam_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, ctx = _make(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_nitroslam_podman_cmd_mounts_cudevshr_cache_when_hami_active(tmp_path: Path) -> None:
    algo, ctx = _make(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    # mid-run cap mutation file mounted onto /tmp/cudevshr.cache:rw
    assert any(HAMI_SHARED_CACHE in part and part.endswith(":rw") for part in spec.cmd)


def test_nitroslam_podman_cmd_includes_cdi_devices_when_session_provides_them(tmp_path: Path) -> None:
    algo, ctx = _make(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_nitroslam_podman_always_attaches_gpu_without_hami_controls(tmp_path: Path) -> None:
    algo, ctx = _make(tmp_path)
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "LD_PRELOAD" not in joined
    assert "libvgpu.so" not in joined
    assert "CUDA_DEVICE_MEMORY_LIMIT" not in joined
    # Nitro always needs CUDA: the GPU device is attached even without HAMi.
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_nitroslam_docker_runtime_uses_gpus_flag_not_cdi(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = NitroSLAMAlgorithm(container_runtime="docker")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--gpus" in spec.cmd
    assert "all" in spec.cmd
    assert "nvidia.com/gpu=all" not in spec.cmd
    assert spec.target_kind == "docker_container"
