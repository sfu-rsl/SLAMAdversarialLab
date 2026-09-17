"""Tests for DROID-SLAM dual-runtime (conda default + podman opt-in) wrapper."""

from pathlib import Path

import pytest

from slamadversariallab.algorithms.droidslam import DROIDSLAMAlgorithm
from slamadversariallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode


def _build_request(tmp_path: Path) -> SLAMRunRequest:
    # Use a "freiburg1" sequence name so the wrapper's calibration lookup
    # picks up TUM_CALIBRATIONS["freiburg1"] without env detection.
    dataset_path = tmp_path / "freiburg1_desk"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    image_dir = dataset_path / "rgb"
    image_dir.mkdir()
    # Wrapper's _prepare_dataset requires at least one PNG/JPG.
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
    image_dir = Path(request.extras["camera_paths"]["left"])
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "slam_config": "tum1",
        "output_dir": request.output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "camera_paths": request.extras["camera_paths"],
        "timestamps_by_frame": request.extras["timestamps_by_frame"],
    }
    return ctx


def test_droidslam_default_constructor_is_conda() -> None:
    algo = DROIDSLAMAlgorithm()
    assert algo.container_runtime is None
    assert algo.runtime_stress_target_kind == "host_process_group"
    assert algo.docker_image == "droidslam:latest"


def test_droidslam_podman_runtime_selection() -> None:
    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    assert algo.container_runtime == "podman"
    assert algo.runtime_stress_target_kind == "podman_container"


def test_droidslam_invalid_runtime_raises() -> None:
    with pytest.raises(ValueError, match="container_runtime must be None or 'podman'"):
        DROIDSLAMAlgorithm(container_runtime="docker")


def test_droidslam_conda_execution_spec_uses_custom_runner(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)

    algo = DROIDSLAMAlgorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is not None
    assert spec.cmd == ["droidslam"]
    # Default for conda spec: target_kind is the base-class default (None).
    assert spec.target_kind in (None, "host_process_group")


def test_droidslam_podman_execution_spec_returns_cmd(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)

    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    assert spec.custom_runner is None
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert spec.target_metadata["container_name"].startswith("droidslam-freiburg1_desk-")
    assert "io_target_paths" in spec.target_metadata

    assert spec.cmd[0] == "podman"
    assert spec.cmd[1] == "run"
    assert "--rm" in spec.cmd
    assert "--name" in spec.cmd

    joined = " ".join(spec.cmd)
    image_dir = Path(request.extras["camera_paths"]["left"])
    assert f"{image_dir.resolve()}:/dataset:ro" in joined
    assert f"{request.output_dir.resolve()}:/output" in joined
    assert "/root/.cache/torch/hub" in joined
    # Calibration is bind-mounted read-only at /calib/calib.txt
    assert ":/calib/calib.txt:ro" in joined
    assert "droidslam:latest" in spec.cmd

    # demo.py invocation tail
    assert spec.cmd[-3] == "bash"
    assert spec.cmd[-2] == "-c"
    main_cmd = spec.cmd[-1]
    assert "python demo.py" in main_cmd
    assert "--imagedir /dataset" in main_cmd
    assert "--calib /calib/calib.txt" in main_cmd
    assert "--weights droid.pth" in main_cmd
    # TUM is stride 1: every frame reaches the SLAM and the deadline does all
    # the dropping. It was 2, which paced 250 every-other frames at the full
    # 30 fps and so made the deadline twice as strict here as for the
    # stride-1 systems. See DROIDSLAMAlgorithm._resolve_stride.
    assert "--stride 1" in main_cmd  # TUM
    assert "--disable_vis" in main_cmd
    assert "--reconstruction_path /output/reconstruction.pth" in main_cmd


def test_droidslam_podman_target_metadata_container_name_is_sanitized(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)

    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    name = spec.target_metadata["container_name"]
    assert name == name.lower()
    assert " " not in name
    assert len(name) <= 120


def test_droidslam_podman_io_target_paths_includes_dataset_and_output(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    ctx = _build_context(request)

    algo = DROIDSLAMAlgorithm(container_runtime="podman")
    spec = algo._build_execution_spec(request, ctx)

    image_dir = Path(request.extras["camera_paths"]["left"])
    paths = spec.target_metadata["io_target_paths"]
    assert str(image_dir.resolve()) in paths
    assert str(request.output_dir.resolve()) in paths
