"""Tests for VGGT-SLAM dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path
from typing import Any, Dict

import pytest

from slamadversariallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode
from slamadversariallab.algorithms.vggtslam import VGGTSLAMAlgorithm


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
    return ctx


def test_vggtslam_default_constructor_is_conda() -> None:
    algo = VGGTSLAMAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"


def test_vggtslam_podman_runtime_selection() -> None:
    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_vggtslam_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        VGGTSLAMAlgorithm(container_runtime="docker")


def test_vggtslam_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)

    algo = VGGTSLAMAlgorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    assert spec.cmd == ["vggtslam"]
    assert spec.target_kind == "host_process_group"


def test_vggtslam_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)

    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert "container_name" in spec.target_metadata
    assert spec.target_metadata["container_name"].startswith("vggtslam-v1_01_easy-")

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd

    joined = " ".join(spec.cmd)
    assert f"{image_folder.resolve()}:/dataset:ro" in joined
    assert f"{request.output_dir.resolve()}:/output" in joined
    assert "/root/.cache/torch/hub" in joined
    assert "vggtslam:latest" in spec.cmd

    # main.py invocation is tail: [..., "vggtslam:latest", "bash", "-c", "<python cmd>"]
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    assert "python main.py" in spec.cmd[-1]
    assert "--image_folder /dataset" in spec.cmd[-1]
    assert "--log_path /output/poses_raw.txt" in spec.cmd[-1]


def test_vggtslam_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    image_folder = tmp_path / "images"
    image_folder.mkdir()
    ctx = _build_context(request, image_folder)

    algo = VGGTSLAMAlgorithm(container_runtime="podman")
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    # No uppercase, no spaces; only [a-z0-9_.-]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120
