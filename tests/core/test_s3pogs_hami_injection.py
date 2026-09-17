"""Tests for HAMi env/mount/device injection into S3PO-GS Podman commands."""

from pathlib import Path
from typing import Any, Dict

from slamadversariallab.algorithms.s3pogs import S3POGSAlgorithm
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


def _s3pogs_with_fake_tree(tmp_path: Path) -> tuple[S3POGSAlgorithm, Path]:
    s3pogs_root = tmp_path / "S3PO-GS"
    cfg_dir = s3pogs_root / "configs" / "mono" / "KITTI"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "04.yaml").write_text(
        'inherit_from: "configs/mono/KITTI/base_config.yaml"\n'
        "\n"
        "Dataset:\n"
        '  dataset_path: "datasets/KITTI/04/"\n',
        encoding="utf-8",
    )
    (cfg_dir / "base_config.yaml").write_text(
        "Results:\n"
        '  save_dir: "results"\n',
        encoding="utf-8",
    )
    algo = S3POGSAlgorithm(container_runtime="podman")
    algo.s3pogs_path = s3pogs_root
    return algo, s3pogs_root


def _build_request_and_context(
    tmp_path: Path,
    s3pogs_root: Path,
) -> tuple[SLAMRunRequest, SLAMRuntimeContext]:
    real_image_dir = tmp_path / "real_left_camera"
    real_image_dir.mkdir(parents=True, exist_ok=True)
    (real_image_dir / "000000.png").write_bytes(b"")

    staged_root = s3pogs_root / "datasets" / "KITTI" / "04"
    staged_root.mkdir(parents=True, exist_ok=True)
    rgb_link = staged_root / "rgb"
    if rgb_link.exists() or rgb_link.is_symlink():
        rgb_link.unlink()
    rgb_link.symlink_to(real_image_dir.resolve())
    (staged_root / "calib.txt").write_text("P0: 0\n", encoding="utf-8")
    (staged_root / "poses.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    request = SLAMRunRequest(
        dataset_path=staged_root,
        slam_config="04",
        output_dir=output_dir,
        dataset_type="kitti",
        sensor_mode=SensorMode.MONO,
        sequence_name="04",
        extras={
            "camera_paths": {"left": str(real_image_dir)},
            "timestamps_by_frame": {0: 0.0},
        },
    )
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="04",
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_root,
    )
    config_file = s3pogs_root / "configs" / "mono" / "KITTI" / "04.yaml"
    ctx.execution_inputs = {
        "prepared_path": staged_root,
        "output_dir": output_dir,
        "is_stereo": False,
        "config_file": config_file,
        "sequence_name": "04",
    }
    ctx.runtime_stress = object()
    return request, ctx


def _build_spec_with_session(algo: S3POGSAlgorithm, ctx: SLAMRuntimeContext, session):
    """Drive ``_build_execution_spec`` with a fake runtime-stress session attached."""
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = session
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def test_s3pogs_podman_cmd_includes_hami_env_vars_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
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


def test_s3pogs_podman_cmd_includes_libvgpu_mount_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "/opt/hami/libvgpu.so:/opt/hami/libvgpu.so:ro" in spec.cmd


def test_s3pogs_podman_cmd_includes_cdi_devices_when_gpu_controls_set(tmp_path: Path) -> None:
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
        devices=["nvidia.com/gpu=all"],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    assert "--device" in spec.cmd
    assert "nvidia.com/gpu=all" in spec.cmd


def test_s3pogs_podman_cmd_unchanged_when_no_gpu_controls(tmp_path: Path) -> None:
    """No HAMi env/mounts unless a session asks for them, but GPU is always attached."""
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
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


def test_s3pogs_podman_cmd_always_mounts_torch_hub_cache(tmp_path: Path) -> None:
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
    session = _FakeSession(env={}, mounts=[])

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/torch/hub" in joined


def test_s3pogs_podman_cmd_always_mounts_huggingface_cache(tmp_path: Path) -> None:
    """MASt3R weights are pulled from HuggingFace by mast3r.model.from_pretrained
    on every run (slam.py:294-295); without a persistent HF cache mount the
    ~2.5 GB weights re-download into the ephemeral container layer on every
    invocation."""
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
    session = _FakeSession(env={}, mounts=[])

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/root/.cache/huggingface" in joined


def test_s3pogs_podman_cmd_includes_hami_cache_bind_mount_when_hami_active(tmp_path: Path) -> None:
    """When LD_PRELOAD is set, the per-container HAMi cache file must be
    bind-mounted onto /tmp/cudevshr.cache:rw so GpuHamiController.apply()
    can write directly to it from the host."""
    algo, _ = _s3pogs_with_fake_tree(tmp_path)
    _request, ctx = _build_request_and_context(tmp_path, algo.s3pogs_path)
    session = _FakeSession(
        env={"LD_PRELOAD": "/opt/hami/libvgpu.so"},
        mounts=[("/opt/hami/libvgpu.so", "/opt/hami/libvgpu.so", "ro")],
    )

    spec = _build_spec_with_session(algo, ctx, session)

    assert spec is not None
    joined = " ".join(spec.cmd)
    assert "/tmp/cudevshr.cache:rw" in joined
