"""Tests for HAMi env/mount/device injection into Photo-SLAM Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.photoslam import PhotoSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


class _FakeSession:
    """Minimal stand-in for ``RuntimeStressOrchestrator`` exposing gpu_launch_config."""

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


def _photoslam_with_fake_cfg_tree(tmp_path: Path) -> tuple[PhotoSLAMAlgorithm, Path]:
    photoslam_root = tmp_path / "Photo-SLAM"
    cfg_tum = photoslam_root / "cfg" / "ORB_SLAM3" / "Monocular" / "TUM"
    cfg_gauss = photoslam_root / "cfg" / "gaussian_mapper" / "Monocular" / "TUM"
    bin_dir = photoslam_root / "bin"
    cfg_tum.mkdir(parents=True, exist_ok=True)
    cfg_gauss.mkdir(parents=True, exist_ok=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "tum_mono").write_text("#!/bin/bash\n", encoding="utf-8")
    (cfg_tum / "tum_freiburg1_desk.yaml").write_text("dummy: true\n", encoding="utf-8")
    (cfg_gauss / "tum_freiburg1_desk.yaml").write_text("dummy: true\n", encoding="utf-8")
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root
    return algo, photoslam_root


def _build_request_and_context(tmp_path: Path) -> tuple[SLAMRunRequest, SLAMRuntimeContext]:
    dataset_path = tmp_path / "freiburg1_desk"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    rgb_dir = dataset_path / "rgb"
    rgb_dir.mkdir()
    (rgb_dir / "0.png").write_bytes(b"")

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="tum_freiburg1_desk",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={
            "camera_paths": {"left": str(rgb_dir)},
            "timestamps_by_frame": {0: 0.0, 1: 0.0333},
        },
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="tum_freiburg1_desk",
        sequence_name=request.sequence_name,
        effective_dataset_path=request.dataset_path,
    )
    ctx.execution_inputs = {
        "dataset_path": dataset_path,
        "slam_config": "tum_freiburg1_desk",
        "output_dir": output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "is_external": False,
        "staged_association_file": None,
        "staged_timestamps_file": None,
    }
    ctx.runtime_stress = object()
    return request, ctx


def _build_spec_with_session(algo: PhotoSLAMAlgorithm, ctx: SLAMRuntimeContext, session):
    """Drive ``_build_execution_spec`` with a fake runtime-stress session attached."""
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = session
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def test_photoslam_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
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
    assert "-e" in spec.cmd
    assert "LD_PRELOAD=/opt/hami/libvgpu.so" in spec.cmd
    assert "CUDA_DEVICE_MEMORY_LIMIT=4096m" in spec.cmd
    assert "CUDA_DEVICE_SM_LIMIT=100" in spec.cmd


def test_photoslam_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_photoslam_podman_cmd_includes_cdi_devices_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_photoslam_podman_cmd_unchanged_when_no_gpu_controls(tmp_path: Path) -> None:
    """No HAMi env/mounts unless a session asks for them, but GPU is always attached."""
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
    session = _FakeSession(env={}, mounts=[])

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "LD_PRELOAD" not in joined
    assert "libvgpu.so" not in joined
    assert "CUDA_DEVICE_MEMORY_LIMIT" not in joined
    assert "--device" in spec.cmd
    # Default device when extras supply none is the CDI nvidia.com/gpu=all spec.
    assert "nvidia.com/gpu=all" in spec.cmd


def test_photoslam_podman_cmd_always_mounts_torch_hub_cache(tmp_path: Path) -> None:
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
    session = _FakeSession(env={}, mounts=[])

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/torch/hub" in joined


def test_photoslam_podman_cmd_includes_hami_cache_bind_mount_when_hami_active(tmp_path: Path) -> None:
    """When LD_PRELOAD is set, the per-container HAMi cache file must be
    bind-mounted onto /tmp/cudevshr.cache:rw so GpuHamiController.apply()
    can write directly to it from the host."""
    algo, _ = _photoslam_with_fake_cfg_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/tmp/cudevshr.cache:rw" in joined
