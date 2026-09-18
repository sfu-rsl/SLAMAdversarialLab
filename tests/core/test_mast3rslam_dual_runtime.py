"""Tests for MASt3R-SLAM dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path

import pytest

from slamadversariallab.algorithms.mast3rslam import MASt3RSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _mast3rslam_with_fake_tree(tmp_path: Path) -> tuple[MASt3RSLAMAlgorithm, Path]:
    """Build a MASt3RSLAMAlgorithm pointed at an isolated fake source tree.

    Lets dual-runtime tests run without the real MASt3R-SLAM submodule on
    disk, mirroring the Photo-SLAM dual-runtime test setup.
    """
    mast3r_root = tmp_path / "MASt3R-SLAM"
    cfg_dir = mast3r_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "base.yaml").write_text("use_calib: False\n", encoding="utf-8")
    (mast3r_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (mast3r_root / "checkpoints" / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth").write_bytes(b"")
    algo = MASt3RSLAMAlgorithm()
    algo.mast3r_path = mast3r_root
    return algo, mast3r_root


def _build_staged_request_and_context(
    tmp_path: Path,
    mast3r_root: Path,
) -> tuple[SLAMRunRequest, SLAMRuntimeContext, Path, Path]:
    """Build a MASt3R-SLAM request with a staged TUM root mimicking
    ``_prepare_dataset``: ``stage_root/tum/rgbd_dataset_freiburg1_<seq>/rgb``
    is a symlink to a real image dir under tmp_path.
    """
    real_rgb_dir = tmp_path / "real_rgb"
    real_rgb_dir.mkdir(parents=True, exist_ok=True)
    (real_rgb_dir / "0.png").write_bytes(b"")
    (real_rgb_dir / "1.png").write_bytes(b"")

    stage_root = tmp_path / "stage_root"
    staged_dataset_path = (
        stage_root / "tum" / "rgbd_dataset_freiburg1_freiburg1_desk"
    )
    staged_dataset_path.mkdir(parents=True, exist_ok=True)
    (staged_dataset_path / "rgb").symlink_to(real_rgb_dir.resolve(), target_is_directory=True)
    (staged_dataset_path / "rgb.txt").write_text(
        "0.0 rgb/0.png\n0.0333 rgb/1.png\n", encoding="utf-8"
    )

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
            "timestamps_by_frame": {0: 0.0, 1: 0.0333},
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
    return request, ctx, staged_dataset_path, real_rgb_dir


def test_mast3rslam_default_constructor_is_conda() -> None:
    algo = MASt3RSLAMAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"
    assert algo.docker_image == "mast3r-slam:latest"


def test_mast3rslam_podman_runtime_selection() -> None:
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"
    assert algo.docker_image == "mast3r-slam:latest"


def test_mast3rslam_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        MASt3RSLAMAlgorithm(container_runtime="docker")


def test_mast3rslam_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    request, ctx, _staged, _real_rgb = _build_staged_request_and_context(tmp_path, mast3r_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    assert spec.cmd == ["mast3rslam"]
    # Default for conda spec: target_kind is the base-class default
    # (None on dataclass init, normalized to "host_process_group" on
    # explicit query).
    assert spec.target_kind in (None, "host_process_group")


def test_mast3rslam_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root
    request, ctx, staged, _real_rgb = _build_staged_request_and_context(tmp_path, mast3r_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("mast3rslam-freiburg1_desk-")
    assert "io_target_paths" in spec.target_metadata

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd
    # MASt3R-SLAM's SharedKeyframes/SharedStates use multiprocessing.Manager
    # backed by /dev/shm; Podman's default 64 MB shm is too small and the
    # process dies with SIGBUS (exit 135) right after config print.
    assert "--shm-size=8g" in spec.cmd

    joined = " ".join(spec.cmd)
    # Staged TUM root → /dataset/tum/rgbd_dataset_freiburg1_data read-only.
    # The container path must carry 'tum' + 'freiburg{N}' segments so
    # MASt3R-SLAM's load_dataset() picks TUMDataset, not the RGBFiles
    # fallback.
    assert f"{staged.resolve()}:/dataset/tum/rgbd_dataset_freiburg1_data:ro" in joined
    # Output dir → /mast3r-slam/logs (MASt3R writes the trajectory here).
    assert f"{request.output_dir.resolve()}:/mast3r-slam/logs" in joined
    # Checkpoints bind-mounted read-only (foundation-model weights).
    assert f"{(mast3r_root / 'checkpoints').resolve()}:/mast3r-slam/checkpoints:ro" in joined
    assert "/root/.cache/torch/hub" in joined
    assert "mast3r-slam:latest" in spec.cmd

    # main.py invocation tail.
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    main_cmd = spec.cmd[-1]
    assert "cd /mast3r-slam" in main_cmd
    assert "python main.py" in main_cmd
    assert "--dataset /dataset/tum/rgbd_dataset_freiburg1_data" in main_cmd
    # The config is COPYed into the image at /mast3r-slam/config/.
    assert "--config /mast3r-slam/config/base.yaml" in main_cmd
    assert "--no-viz" in main_cmd


def test_mast3rslam_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root
    request, ctx, _staged, _real_rgb = _build_staged_request_and_context(tmp_path, mast3r_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


def test_mast3rslam_podman_io_target_paths_includes_dataset_and_output(tmp_path: Path) -> None:
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root
    request, ctx, staged, _real_rgb = _build_staged_request_and_context(tmp_path, mast3r_root)

    spec = algo._build_execution_spec(request, ctx)

    paths = spec.target_metadata["io_target_paths"]
    assert str(staged.resolve()) in paths
    assert str(request.output_dir.resolve()) in paths


def test_mast3rslam_podman_cmd_bind_mounts_absolute_symlink_targets(tmp_path: Path) -> None:
    """The staged TUM root contains an absolute-path symlink for rgb/.
    The wrapper must bind-mount the symlink target at its same host path
    inside the container so the symlink resolves. Otherwise MASt3R-SLAM
    would see a broken symlink for rgb/ and fail to read frames."""
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root
    request, ctx, _staged, real_rgb = _build_staged_request_and_context(tmp_path, mast3r_root)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    joined = " ".join(spec.cmd)
    expected_mount = f"{real_rgb.resolve()}:{real_rgb.resolve()}:ro"
    assert expected_mount in joined


def test_mast3rslam_podman_external_config_falls_back_to_host_path(tmp_path: Path) -> None:
    """External configs outside the mast3r_path can't be translated to a
    container path; the wrapper falls back to bind-mounting the file at
    its absolute host path and using that path inside the container."""
    algo, mast3r_root = _mast3rslam_with_fake_tree(tmp_path)
    algo = MASt3RSLAMAlgorithm(container_runtime="podman")
    algo.mast3r_path = mast3r_root

    external_root = tmp_path / "external_cfg"
    external_root.mkdir()
    external_cfg = external_root / "my_external.yaml"
    external_cfg.write_text("use_calib: False\n", encoding="utf-8")

    # Direct unit call so we can verify the helper in isolation.
    assert algo._translate_config_path_to_container(external_cfg) == str(external_cfg.resolve())
    assert algo._translate_config_path_to_container(mast3r_root / "config" / "base.yaml") == (
        "/mast3r-slam/config/base.yaml"
    )
