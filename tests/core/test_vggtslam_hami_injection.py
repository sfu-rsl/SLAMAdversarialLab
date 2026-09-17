"""Tests for HAMi env/mount/device injection into VGGT-SLAM Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode
from slamadversariallab.algorithms.vggtslam import VGGTSLAMAlgorithm


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
        slam_config="V1_01_easy",
        output_dir=output_dir,
        dataset_type="euroc",
        sensor_mode=SensorMode.MONO,
        sequence_name="V1_01_easy",
    )


def _build_context(request: SLAMRunRequest, image_folder: Path) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name=None,
        sequence_name=request.sequence_name,
        effective_dataset_path=image_folder,
    )
    ctx.execution_inputs = {
        "image_folder": image_folder,
        "output_dir": request.output_dir,
        "is_stereo": False,
        "output_poses": request.output_dir / "poses_raw.txt",
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


def test_vggtslam_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
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


def test_vggtslam_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_vggtslam_podman_cmd_includes_cdi_devices_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_vggtslam_podman_cmd_unchanged_when_no_gpu_controls(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "LD_PRELOAD" not in joined
    assert "libvgpu.so" not in joined
    assert "CUDA_DEVICE_MEMORY_LIMIT" not in joined
    # VGGT-SLAM always needs CUDA, so the GPU device is always attached even
    # without HAMi controls. Default is "nvidia.com/gpu=all".
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_vggtslam_podman_cmd_always_mounts_torch_hub_cache(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/torch/hub" in joined
