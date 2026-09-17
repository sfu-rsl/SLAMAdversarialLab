"""Tests for the Nitro-SLAM EuRoC stereo-inertial command and IMU staging."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from slamadversariallab.algorithms.nitroslam import NitroSLAMAlgorithm
from slamadversariallab.algorithms.types import (
    SLAMRunRequest,
    SLAMRuntimeContext,
    SensorMode,
)


class _FakeSession:
    def gpu_launch_config(self) -> Dict[str, Any]:
        return {"env": {}, "mounts": [], "devices": []}


def _build_request(tmp_path: Path) -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="EuRoC.yaml",
        output_dir=output_dir,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
    )


def _build_context(
    request: SLAMRunRequest,
    staged_path: Path,
    image_mounts: Optional[List[Tuple[str, str]]] = None,
) -> SLAMRuntimeContext:
    ctx = SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="EuRoC.yaml",
        sequence_name=request.sequence_name,
        effective_dataset_path=staged_path,
    )
    ctx.execution_inputs = {
        "dataset_path": staged_path,
        "output_dir": request.output_dir,
        "euroc_image_mounts": list(image_mounts or []),
    }
    ctx.runtime_stress = object()
    return ctx


def _build_spec(algo, ctx):
    algo._active_runtime_context = ctx
    ctx.runtime_stress_session = _FakeSession()
    try:
        return algo._build_execution_spec(ctx.request, ctx)
    finally:
        algo._active_runtime_context = None


def _bash_command(spec) -> str:
    # cmd = [runtime, run, ..., image, "bash", "-c", <bash_cmd>]
    assert spec.cmd[-2] == "-c"
    return spec.cmd[-1]


def test_nitroslam_command_uses_stereo_inertial_binary_with_module_args(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    ctx = _build_context(request, staged)
    algo = NitroSLAMAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    bash_cmd = _bash_command(spec)

    assert "./Examples/Stereo-Inertial/stereo_inertial_euroc" in bash_cmd
    assert "Vocabulary/ORBvoc.txt" in bash_cmd
    assert "Examples/Stereo-Inertial/EuRoC.yaml" in bash_cmd
    # Single-sequence stereo_inertial_euroc REQUIRES a trajectory_file_name; the
    # full arg order is: ... sequence times_file file_name statsDir <7 module args>.
    assert (
        "/dataset /dataset/orbslam3_timestamps.txt nitro /output 1 1 1 11110 1111 11111"
        in bash_cmd
    )
    # f_<name>.txt / kf_<name>.txt are renamed to Camera/KeyFrameTrajectory.txt
    # so the inherited trajectory collection applies.
    assert "cp f_nitro.txt /output/CameraTrajectory.txt" in bash_cmd
    assert "cp kf_nitro.txt /output/KeyFrameTrajectory.txt" in bash_cmd


def test_nitroslam_command_metadata_and_mounts(tmp_path: Path) -> None:
    request = _build_request(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    image_mounts = [
        (str(tmp_path / "cam0"), "/dataset/mav0/cam0/data"),
        (str(tmp_path / "cam1"), "/dataset/mav0/cam1/data"),
    ]
    ctx = _build_context(request, staged, image_mounts=image_mounts)
    algo = NitroSLAMAlgorithm(container_runtime="podman")

    spec = _build_spec(algo, ctx)
    assert spec is not None
    assert spec.log_prefix == "Nitro-SLAM"
    assert spec.target_kind == "podman_container"
    assert spec.target_metadata is not None
    assert "container_name" in spec.target_metadata
    assert spec.target_metadata["container_name"].startswith("nitroslam-")

    joined = " ".join(spec.cmd)
    assert f"{tmp_path / 'cam0'}:/dataset/mav0/cam0/data:ro" in joined
    assert f"{tmp_path / 'cam1'}:/dataset/mav0/cam1/data:ro" in joined
    assert f"{staged.resolve()}:/dataset:ro" in joined


def _imu_request(tmp_path: Path, with_imu: bool) -> SLAMRunRequest:
    dataset_path = tmp_path / "euroc"
    if with_imu:
        imu_dir = dataset_path / "mav0" / "imu0"
        imu_dir.mkdir(parents=True)
        (imu_dir / "data.csv").write_text("#ts,wx,wy,wz,ax,ay,az\n1,0,0,0,0,0,9.8\n")
        (imu_dir / "sensor.yaml").write_text("sensor_type: imu\n")
    else:
        dataset_path.mkdir(parents=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="EuRoC.yaml",
        output_dir=tmp_path / "out",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
    )


def test_nitroslam_stages_euroc_imu_stream(tmp_path: Path) -> None:
    request = _imu_request(tmp_path, with_imu=True)
    staged = tmp_path / "staged"
    staged.mkdir()
    algo = NitroSLAMAlgorithm(container_runtime="podman")

    algo._stage_euroc_imu(request, staged)

    assert (staged / "mav0" / "imu0" / "data.csv").exists()
    assert (staged / "mav0" / "imu0" / "sensor.yaml").exists()


def test_nitroslam_resolve_imu_returns_none_when_missing(tmp_path: Path) -> None:
    request = _imu_request(tmp_path, with_imu=False)
    algo = NitroSLAMAlgorithm(container_runtime="podman")
    assert algo._resolve_euroc_imu_csv(request) is None


def test_nitroslam_stage_imu_raises_when_missing(tmp_path: Path) -> None:
    request = _imu_request(tmp_path, with_imu=False)
    staged = tmp_path / "staged"
    staged.mkdir()
    algo = NitroSLAMAlgorithm(container_runtime="podman")
    with pytest.raises(RuntimeError):
        algo._stage_euroc_imu(request, staged)
