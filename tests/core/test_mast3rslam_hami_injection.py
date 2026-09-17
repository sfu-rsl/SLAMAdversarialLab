"""Tests for HAMi env/mount/device injection into MASt3R-SLAM Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.mast3rslam import MASt3RSLAMAlgorithm
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


def _mast3rslam_with_fake_tree(tmp_path: Path) -> tuple[MASt3RSLAMAlgorithm, Path]:
    mast3r_root = tmp_path / "MASt3R-SLAM"
    cfg_dir = mast3r_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "base.yaml").write_text("use_calib: False\n", encoding="utf-8")
    (mast3r_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (mast3r_root / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").write_bytes(b"")
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root
    return algo, mast3r_root


def _build_staged_request_and_context(
    tmp_path: Path,
    mast3r_root: Path,
) -> tuple[SLAMRunRequest, SLAMRuntimeContext]:
    real_rgb_dir = tmp_path / "real_rgb"
    real_rgb_dir.mkdir(parents=True, exist_ok=True)
    (real_rgb_dir / "0.png").write_bytes(b"")

    stage_root = tmp_path / "stage_root"
    staged_dataset_path = (
        stage_root / "tum" / "rgbd_dataset_freiburg1_freiburg1_desk"
    )
    staged_dataset_path.mkdir(parents=True, exist_ok=True)
    (staged_dataset_path / "rgb").symlink_to(real_rgb_dir.resolve(), target_is_directory=True)
    (staged_dataset_path / "rgb.txt").write_text("0.0 rgb/0.png\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    request = SLAMRunRequest(
        dataset_path=stage_root,
        slam_config="base",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={
            "camera_paths": {"left": str(real_rgb_dir)},
            "timestamps_by_frame": {0: 0.0},
        },
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="base",
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_dataset_path,
    )
    ctx.execution_inputs = {
        "output_dir": output_dir,
        "is_stereo": False,
        "prepared_path": staged_dataset_path,
        "config_file": mast3r_root / "config" / "base.yaml",
        "dataset_type": "tum",
        "sequence_name": "freiburg1_desk",
        "log_basenames": [
            "rgbd_dataset_freiburg1_freiburg1_desk",
            "freiburg1_desk",
        ],
    }
    ctx.runtime_stress = object()
    return request, ctx


def _build_spec_with_session(algo: MASt3RSLAMAlgorithm, ctx: SLAMRuntimeContext, session):
    """Drive ``_build_execution_spec`` with a fake runtime-stress session attached."""
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = session
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def test_mast3rslam_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
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


def test_mast3rslam_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_mast3rslam_podman_cmd_includes_cdi_devices_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_mast3rslam_podman_cmd_unchanged_when_no_gpu_controls(tmp_path: Path) -> None:
    """No HAMi env/mounts unless a session asks for them, but GPU is always attached."""
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
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


def test_mast3rslam_podman_cmd_always_mounts_torch_hub_cache(tmp_path: Path) -> None:
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
    session = _FakeSession(env={}, mounts=[])

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/torch/hub" in joined


def test_mast3rslam_podman_cmd_includes_hami_cache_bind_mount_when_hami_active(tmp_path: Path) -> None:
    """When LD_PRELOAD is set, the per-container HAMi cache file must be
    bind-mounted onto /tmp/cudevshr.cache:rw so GpuHamiController.apply()
    can write directly to it from the host."""
    algo, _ = _mast3rslam_with_fake_tree(tmp_path)
    _request, ctx = _build_staged_request_and_context(tmp_path, algo.mast3r_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/tmp/cudevshr.cache:rw" in joined
