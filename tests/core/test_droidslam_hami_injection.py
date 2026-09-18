"""Tests for HAMi env/mount/device injection into DROID-SLAM Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm
from slamadversariallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode


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
    dataset_path = tmp_path / "freiburg1_desk"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    image_dir = dataset_path / "rgb"
    image_dir.mkdir()
    (image_dir / "0.png").write_bytes(b"")

    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="tum1",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={
            "camera_paths": {"left": str(image_dir)},
            "timestamps_by_frame": {0: 0.0, 1: 0.0333},
        },
    )


def _build_context(request: SLAMRunRequest) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name=None,
        sequence_name=request.sequence_name,
        effective_dataset_path=request.dataset_path,
    )
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "slam_config": "tum1",
        "output_dir": request.output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "camera_paths": request.extras["camera_paths"],
        "timestamps_by_frame": request.extras["timestamps_by_frame"],
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


def test_droidslam_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={
            "LD_PRELOAD": "/opt/hami/libvgpu.so",
            "CUDA_DEVICE_MEMORY_LIMIT": "8192m",
            "CUDA_DEVICE_SM_LIMIT": "100",
        },
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "-e" in spec.cmd
    assert "LD_PRELOAD=/opt/hami/libvgpu.so" in spec.cmd
    assert "CUDA_DEVICE_MEMORY_LIMIT=8192m" in spec.cmd
    assert "CUDA_DEVICE_SM_LIMIT=100" in spec.cmd


def test_droidslam_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_droidslam_podman_cmd_includes_cdi_devices_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_droidslam_podman_cmd_unchanged_when_no_gpu_controls(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "LD_PRELOAD" not in joined
    assert "libvgpu.so" not in joined
    assert "CUDA_DEVICE_MEMORY_LIMIT" not in joined
    assert "--device" in spec.cmd
    # GPU device is attached — either via CDI or raw device nodes.
    assert any("nvidia" in tok for tok in spec.cmd)


def test_droidslam_podman_cmd_always_mounts_torch_hub_cache(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/torch/hub" in joined
