"""Tests for HAMi env/mount/device injection into DPVO Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.dpvo import DPVOAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


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


def test_dpvo_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={
            "LD_PRELOAD": "/opt/hami/libvgpu.so",
            "CUDA_DEVICE_MEMORY_LIMIT": "4096m",
            "CUDA_DEVICE_SM_LIMIT": "100",
        },
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "LD_PRELOAD=/opt/hami/libvgpu.so" in spec.cmd
    assert "CUDA_DEVICE_MEMORY_LIMIT=4096m" in spec.cmd
    assert "CUDA_DEVICE_SM_LIMIT=100" in spec.cmd


def test_dpvo_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd
    # HAMi active -> per-container cache file bind-mounted rw for mid-run
    # cap mutation.
    joined = " ".join(spec.cmd)
    assert "cudevshr" in joined


def test_dpvo_podman_cmd_always_has_cdi_device(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")
    # No GPU controls at all: CDI device must still be attached (DPVO always
    # needs CUDA).
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "nvidia.com/gpu=all" in joined


def test_dpvo_realtime_env_propagates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SAL_DEADLINE_FPS", "30")
    monkeypatch.setenv(
        "SAL_RUNTIME_PATH",
        str(Path(__file__).resolve().parents[2] / "src" / "runtime_stress"),
    )
    request = _build_request(tmp_path)
    ctx = _build_context(request)
    algo = DPVOAlgorithm(container_runtime="podman")
    session = _FakeSession(env={}, mounts=[])
    spec = _build_spec_with_session(algo, ctx, session)

    joined = " ".join(spec.cmd)
    assert "-e SAL_DEADLINE_FPS=30" in joined
    assert "-e SAL_DROP_LOG_PATH=/output/deadline_drops.json" in joined
    assert "/sal_runtime" in joined
