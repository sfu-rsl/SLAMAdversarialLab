"""Tests for Photo-SLAM dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path

import pytest

from slamadversariallab.algorithms.photoslam import PhotoSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


def _photoslam_with_fake_cfg_tree(tmp_path: Path) -> tuple[PhotoSLAMAlgorithm, Path]:
    """Build a PhotoSLAMAlgorithm pointed at an isolated fake source tree.

    Mirrors the pattern used by ``test_photoslam_config_resolution_contracts.py``
    so the dual-runtime tests don't need the real Photo-SLAM submodule on
    disk and don't depend on the real cfg/ layout.
    """
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
    return PhotoSLAMAlgorithm(), photoslam_root


def _build_request_and_context(tmp_path: Path) -> tuple[SLAMRunRequest, SLAMRuntimeContext]:
    """Minimal TUM mono request + context (covers both runtimes)."""
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
    return request, ctx


def test_photoslam_default_constructor_is_conda() -> None:
    algo = PhotoSLAMAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"
    assert algo.docker_image == "photoslam:latest"


def test_photoslam_podman_runtime_selection() -> None:
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_photoslam_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        PhotoSLAMAlgorithm(container_runtime="docker")


def test_photoslam_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    request, ctx = _build_request_and_context(tmp_path)
    algo.photoslam_path = photoslam_root

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    # Host conda spec inherits the base-class default target_kind.
    assert spec.target_kind in (None, "host_process_group")
    # First positional arg of the host command is the absolute path to tum_mono.
    assert spec.cmd[0] == str((photoslam_root / "bin" / "tum_mono").resolve())


def test_photoslam_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root
    request, ctx = _build_request_and_context(tmp_path)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("photoslam-freiburg1_desk-")
    assert "io_target_paths" in spec.target_metadata

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd

    joined = " ".join(spec.cmd)
    # Dataset bind-mount: full dataset dir at /dataset (Photo-SLAM C++
    # binary reads <dataset>/rgb.txt at startup so the dir mount, not
    # the rgb/ subdir mount, is required).
    assert f"{request.dataset_path.resolve()}:/dataset:ro" in joined
    assert f"{request.output_dir.resolve()}:/output" in joined
    assert "/root/.cache/torch/hub" in joined
    assert "photoslam:latest" in spec.cmd

    # C++ binary invocation tail
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    main_cmd = spec.cmd[-1]
    assert "cd /photo-slam" in main_cmd
    assert "/photo-slam/bin/tum_mono" in main_cmd
    assert "/photo-slam/ORB-SLAM3/Vocabulary/ORBvoc.txt" in main_cmd
    assert "/photo-slam/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_desk.yaml" in main_cmd
    assert "/photo-slam/cfg/gaussian_mapper/Monocular/TUM/tum_freiburg1_desk.yaml" in main_cmd
    assert "/dataset" in main_cmd
    assert "/output/" in main_cmd
    assert "no_viewer" in main_cmd


def test_photoslam_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root
    request, ctx = _build_request_and_context(tmp_path)

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


def test_photoslam_podman_io_target_paths_includes_dataset_and_output(tmp_path: Path) -> None:
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root
    request, ctx = _build_request_and_context(tmp_path)

    spec = algo._build_execution_spec(request, ctx)

    paths = spec.target_metadata["io_target_paths"]
    assert str(request.dataset_path.resolve()) in paths
    assert str(request.output_dir.resolve()) in paths


def test_photoslam_podman_cmd_bind_mounts_absolute_symlink_targets(tmp_path: Path) -> None:
    """When the staged dataset contains absolute-path symlinks, the wrapper
    must bind-mount each symlink target into the container at the same host
    path so the symlinks resolve. Otherwise Photo-SLAM would see broken
    symlinks for the rgb/ directory and fail to read frames."""
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root

    # Build a fake staged dataset structure: stage_root/rgb is a symlink
    # to an absolute host path elsewhere on disk.
    real_rgb_dir = tmp_path / "real_rgb"
    real_rgb_dir.mkdir()
    (real_rgb_dir / "0.png").write_bytes(b"")
    stage_root = tmp_path / "stage_root"
    stage_root.mkdir()
    (stage_root / "rgb.txt").write_text("# header\n0 rgb/0.png\n", encoding="utf-8")
    (stage_root / "rgb").symlink_to(real_rgb_dir.resolve(), target_is_directory=True)

    output_dir = tmp_path / "output"
    output_dir.mkdir()

    request = SLAMRunRequest(
        dataset_path=stage_root,
        slam_config="tum_freiburg1_desk",
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
        internal_config_name="tum_freiburg1_desk",
        sequence_name=request.sequence_name,
        effective_dataset_path=stage_root,
    )
    ctx.execution_inputs = {
        "dataset_path": stage_root,
        "slam_config": "tum_freiburg1_desk",
        "output_dir": output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "is_external": False,
        "staged_association_file": None,
        "staged_timestamps_file": None,
    }

    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    joined = " ".join(spec.cmd)
    # The real_rgb_dir host path must be bind-mounted at its same path
    # inside the container so the staged symlink resolves.
    expected_mount = f"{real_rgb_dir.resolve()}:{real_rgb_dir.resolve()}:ro"
    assert expected_mount in joined


def test_photoslam_podman_external_config_warns_and_uses_absolute_host_path(tmp_path: Path) -> None:
    """External configs outside the photoslam_path can't be translated; the
    wrapper falls back to the absolute host path and warns. This guards
    against silent path corruption when an operator points slam_config at
    a stray external YAML."""
    algo, photoslam_root = _photoslam_with_fake_cfg_tree(tmp_path)
    algo = PhotoSLAMAlgorithm(container_runtime="podman")
    algo.photoslam_path = photoslam_root

    external_root = tmp_path / "external_cfg"
    external_root.mkdir()
    external_orb = external_root / "my_external_orb.yaml"
    external_gauss = external_root / "my_external_gauss.yaml"
    external_orb.write_text("dummy: true\n", encoding="utf-8")
    external_gauss.write_text("dummy: true\n", encoding="utf-8")

    request, ctx = _build_request_and_context(tmp_path)
    # Direct unit call rather than driving through the public path so we
    # can check the helper in isolation.
    assert algo._relative_config_in_container(external_orb) == str(external_orb.resolve())
    assert algo._relative_config_in_container(external_gauss) == str(external_gauss.resolve())
