"""Tests for algorithm runtime camera-path contract usage."""

import json
import os
import time
from pathlib import Path

import pytest

from slamadverseriallab.algorithms.droidslam import DROIDSLAMAlgorithm
from slamadverseriallab.algorithms.gigaslam import GigaSLAMAlgorithm
from slamadverseriallab.algorithms.photoslam import PhotoSLAMAlgorithm
from slamadverseriallab.algorithms.s3pogs import S3POGSAlgorithm
from slamadverseriallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode
from slamadverseriallab.algorithms.vggtslam import VGGTSLAMAlgorithm


def _build_run_request(
    tmp_path: Path,
    *,
    dataset_type: str,
    sequence_name: str,
    timestamps_by_frame,
) -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset_root"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=output_dir,
        dataset_type=dataset_type,
        sensor_mode=SensorMode.MONO,
        sequence_name=sequence_name,
        extras={"timestamps_by_frame": timestamps_by_frame},
    )


def _build_context(request: SLAMRunRequest) -> SLAMRuntimeContext:
    return SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="dummy_config",
        sequence_name=request.sequence_name,
    )


def test_droid_prepare_dataset_uses_left_camera_path_from_request_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    left_images = dataset_path / "camera_left"
    left_images.mkdir(parents=True, exist_ok=True)
    (left_images / "000000.png").write_bytes(b"x")

    algo = DROIDSLAMAlgorithm()
    resolved = algo._prepare_dataset(
        dataset_path=dataset_path,
        dataset_type="tum",
        camera_paths={"left": str(left_images)},
    )

    assert resolved == left_images


def test_droid_prepare_dataset_fails_without_left_camera_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    algo = DROIDSLAMAlgorithm()
    resolved = algo._prepare_dataset(
        dataset_path=dataset_path,
        dataset_type="tum",
        camera_paths={},
    )

    assert resolved is None


def test_droid_convert_reconstruction_uses_request_timestamp_contract(tmp_path: Path) -> None:
    import torch

    algo = DROIDSLAMAlgorithm()
    reconstruction_path = tmp_path / "reconstruction.pth"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "tstamps": torch.tensor([0, 1], dtype=torch.int64),
        "poses": torch.tensor(
            [
                [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
                [4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
    }
    torch.save(data, reconstruction_path)

    converted = algo._convert_reconstruction(
        reconstruction_path=reconstruction_path,
        output_dir=output_dir,
        dataset_type="tum",
        stride=2,
        timestamps_by_frame={0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0},
    )

    assert converted is True
    lines = (output_dir / "CameraTrajectory.txt").read_text(encoding="utf-8").splitlines()
    assert lines[0].startswith("10.0 ")
    assert lines[1].startswith("12.0 ")


def test_droid_convert_reconstruction_fails_on_out_of_range_timestamp_index(tmp_path: Path) -> None:
    import torch

    algo = DROIDSLAMAlgorithm()
    reconstruction_path = tmp_path / "reconstruction.pth"
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "tstamps": torch.tensor([2], dtype=torch.int64),
        "poses": torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    }
    torch.save(data, reconstruction_path)

    converted = algo._convert_reconstruction(
        reconstruction_path=reconstruction_path,
        output_dir=output_dir,
        dataset_type="tum",
        stride=2,
        timestamps_by_frame={0: 10.0, 1: 11.0, 2: 12.0, 3: 13.0},
    )

    assert converted is False


def test_photoslam_stages_euroc_from_runtime_contracts_without_mutating_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    left_images = dataset_path / "custom_left"
    right_images = dataset_path / "custom_right"
    left_images.mkdir(parents=True, exist_ok=True)
    right_images.mkdir(parents=True, exist_ok=True)
    (left_images / "1000000000.png").write_bytes(b"x")
    (left_images / "1000000001.png").write_bytes(b"x")
    (right_images / "1000000000.png").write_bytes(b"x")
    (right_images / "1000000001.png").write_bytes(b"x")

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
        extras={
            "camera_paths": {
                "left": str(left_images),
                "right": str(right_images),
            },
            "timestamps_by_frame": {
                0: 1000000000,
                1: 1000000001,
            },
        },
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(request)

    algo = PhotoSLAMAlgorithm()
    staged = algo._stage_dataset(request, ctx)

    assert staged is not None
    assert staged != dataset_path
    assert not (dataset_path / "mav0" / "cam0" / "data").exists()
    assert (staged / "mav0" / "cam0" / "data").is_symlink()
    assert (staged / "mav0" / "cam1" / "data").is_symlink()
    assert (staged / "mav0" / "cam0" / "data").resolve() == left_images.resolve()
    assert (staged / "mav0" / "cam1" / "data").resolve() == right_images.resolve()
    assert (staged / "photoslam_timestamps.txt").read_text(encoding="utf-8").splitlines() == [
        "1000000000",
        "1000000001",
    ]

    algo._cleanup_staged_dataset(request, ctx)
    assert not staged.exists()


def test_photoslam_euroc_stage_fails_fast_on_timestamp_image_count_mismatch(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    left_images = dataset_path / "left"
    right_images = dataset_path / "right"
    left_images.mkdir(parents=True, exist_ok=True)
    right_images.mkdir(parents=True, exist_ok=True)
    (left_images / "1000000000.png").write_bytes(b"x")
    (left_images / "1000000001.png").write_bytes(b"x")
    (right_images / "1000000000.png").write_bytes(b"x")
    (right_images / "1000000001.png").write_bytes(b"x")

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
        extras={
            "camera_paths": {
                "left": str(left_images),
                "right": str(right_images),
            },
            "timestamps_by_frame": {
                0: 1000000000,
            },
        },
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(request)

    algo = PhotoSLAMAlgorithm()
    with pytest.raises(RuntimeError, match="timestamps must match stereo image count"):
        algo._stage_dataset(request, ctx)


def test_photoslam_tum_stage_fails_without_left_camera_contract(tmp_path: Path) -> None:
    request = SLAMRunRequest(
        dataset_path=tmp_path / "dataset",
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg1_desk",
        extras={},
    )
    request.dataset_path.mkdir(parents=True, exist_ok=True)
    request.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(request)

    algo = PhotoSLAMAlgorithm()
    with pytest.raises(RuntimeError, match="camera_paths"):
        algo._stage_dataset(request, ctx)


def test_photoslam_tum_rgbd_stage_requires_existing_association_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    left_images = dataset_path / "rgb"
    depth_dir = dataset_path / "depth"
    left_images.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    (left_images / "000000.png").write_bytes(b"x")
    (depth_dir / "000000.png").write_bytes(b"x")

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="tum",
        sensor_mode=SensorMode.RGBD,
        sequence_name="freiburg1_desk",
        extras={"camera_paths": {"left": str(left_images)}},
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(request)

    algo = PhotoSLAMAlgorithm()
    with pytest.raises(RuntimeError, match="requires an existing association file"):
        algo._stage_dataset(request, ctx)


def test_photoslam_stage_failure_cleans_staging_artifacts(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)
    left_images = dataset_path / "left"
    left_images.mkdir(parents=True, exist_ok=True)
    (left_images / "1000000000.png").write_bytes(b"x")

    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=tmp_path / "output",
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        sequence_name="V1_01_easy",
        extras={
            "camera_paths": {"left": str(left_images)},
            "timestamps_by_frame": {0: 1000000000},
        },
    )
    request.output_dir.mkdir(parents=True, exist_ok=True)
    ctx = _build_context(request)

    algo = PhotoSLAMAlgorithm()
    with pytest.raises(RuntimeError, match="camera_paths"):
        algo._stage_dataset(request, ctx)

    assert "photoslam_stage_root" not in ctx.staging_artifacts
    assert "photoslam_association_file" not in ctx.staging_artifacts
    assert "photoslam_timestamps_file" not in ctx.staging_artifacts
    assert algo._staged_dataset_dir is None


def test_photoslam_execution_spec_fails_fast_on_tum_stereo_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_root"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.STEREO,
        sequence_name="freiburg1_desk",
        extras={},
    )
    ctx = _build_context(request)
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "slam_config": "dummy_config",
        "output_dir": request.output_dir,
        "dataset_type": "tum",
        "is_stereo": True,
        "is_external": False,
        "staged_association_file": None,
        "staged_timestamps_file": None,
    }

    algo = PhotoSLAMAlgorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is None


def test_photoslam_execution_spec_fails_fast_on_euroc_non_stereo_mode(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset_root"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    request = SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=output_dir,
        dataset_type="euroc",
        sensor_mode=SensorMode.MONO,
        sequence_name="V1_01_easy",
        extras={},
    )
    ctx = _build_context(request)
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "slam_config": "dummy_config",
        "output_dir": request.output_dir,
        "dataset_type": "euroc",
        "is_stereo": False,
        "is_external": False,
        "staged_association_file": None,
        "staged_timestamps_file": None,
    }

    algo = PhotoSLAMAlgorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is None


def test_vggtslam_prepare_euroc_dataset_uses_left_camera_path_from_request_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    left_images = dataset_path / "custom_left"
    left_images.mkdir(parents=True, exist_ok=True)
    (left_images / "000000.png").write_bytes(b"x")

    algo = VGGTSLAMAlgorithm()
    resolved = algo._prepare_euroc_dataset(
        dataset_path=dataset_path,
        camera_paths={"left": str(left_images)},
    )

    assert resolved == left_images


def test_vggtslam_prepare_euroc_dataset_fails_without_left_camera_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    algo = VGGTSLAMAlgorithm()
    resolved = algo._prepare_euroc_dataset(
        dataset_path=dataset_path,
        camera_paths={},
    )

    assert resolved is None


def test_gigaslam_resolves_left_camera_path_from_request_contract(tmp_path: Path) -> None:
    left_images = tmp_path / "left_images"
    left_images.mkdir(parents=True, exist_ok=True)
    (left_images / "000000.png").write_bytes(b"x")

    algo = GigaSLAMAlgorithm()
    resolved = algo._resolve_left_camera_path({"left": str(left_images)})

    assert resolved == left_images


def test_gigaslam_fails_without_left_camera_contract() -> None:
    algo = GigaSLAMAlgorithm()
    resolved = algo._resolve_left_camera_path({})

    assert resolved is None


def test_s3pogs_prepare_dataset_uses_left_camera_path_for_rgb_symlink(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    left_images = dataset_path / "custom_left"
    left_images.mkdir(parents=True, exist_ok=True)
    (left_images / "000000.png").write_bytes(b"x")
    (dataset_path / "poses.txt").write_text("0 0 0 0 0 0 0 0 0 0 0 0\n", encoding="utf-8")

    algo = S3POGSAlgorithm()
    algo.s3pogs_path = tmp_path / "S3PO-GS"

    prepared = algo._prepare_dataset(
        dataset_path=dataset_path,
        sequence="4",
        camera_paths={"left": str(left_images)},
    )

    assert prepared is not None
    rgb_link = prepared / "rgb"
    assert rgb_link.is_symlink()
    assert rgb_link.resolve() == left_images.resolve()

    algo._cleanup_temp_dataset_link()


def test_s3pogs_prepare_dataset_fails_without_left_camera_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir(parents=True, exist_ok=True)

    algo = S3POGSAlgorithm()
    algo.s3pogs_path = tmp_path / "S3PO-GS"

    prepared = algo._prepare_dataset(
        dataset_path=dataset_path,
        sequence="4",
        camera_paths={},
    )

    assert prepared is None


def test_gigaslam_conversion_uses_request_timestamp_contract(tmp_path: Path) -> None:
    algo = GigaSLAMAlgorithm()
    request = _build_run_request(
        tmp_path,
        dataset_type="kitti",
        sequence_name="04",
        timestamps_by_frame={2: 12.5},
    )
    ctx = _build_context(request)

    raw_traj = tmp_path / "poses_est.txt"
    raw_traj.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")
    (tmp_path / "poses_idx.txt").write_text("2\n", encoding="utf-8")

    converted = algo._convert_raw_trajectory_to_tum(raw_traj, request, ctx)

    assert converted is not None
    line = converted.read_text(encoding="utf-8").strip()
    assert line.startswith("12.5 ")


def test_gigaslam_conversion_fails_when_timestamp_missing(tmp_path: Path) -> None:
    algo = GigaSLAMAlgorithm()
    request = _build_run_request(
        tmp_path,
        dataset_type="kitti",
        sequence_name="04",
        timestamps_by_frame={},
    )
    ctx = _build_context(request)

    raw_traj = tmp_path / "poses_est.txt"
    raw_traj.write_text("1 0 0 0 0 1 0 0 0 0 1 0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing timestamps_by_frame"):
        algo._convert_raw_trajectory_to_tum(raw_traj, request, ctx)


def test_s3pogs_conversion_uses_request_timestamp_contract(tmp_path: Path) -> None:
    algo = S3POGSAlgorithm()
    request = _build_run_request(
        tmp_path,
        dataset_type="kitti",
        sequence_name="04",
        timestamps_by_frame={5: 1.25},
    )
    ctx = _build_context(request)

    raw_traj = tmp_path / "trj_final.json"
    raw_traj.write_text(
        json.dumps(
            {
                "trj_id": [5],
                "trj_est": [
                    [
                        [1.0, 0.0, 0.0, 0.1],
                        [0.0, 1.0, 0.0, 0.2],
                        [0.0, 0.0, 1.0, 0.3],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ],
            }
        ),
        encoding="utf-8",
    )

    converted = algo._convert_raw_trajectory_to_tum(raw_traj, request, ctx)

    assert converted is not None
    line = converted.read_text(encoding="utf-8").strip()
    assert line.startswith("1.25 ")


def test_s3pogs_find_raw_trajectory_skips_lookup_when_execution_failed(tmp_path: Path) -> None:
    algo = S3POGSAlgorithm()
    algo.s3pogs_path = tmp_path / "S3PO-GS"
    request = _build_run_request(
        tmp_path,
        dataset_type="kitti",
        sequence_name="04",
        timestamps_by_frame={0: 0.0},
    )
    ctx = _build_context(request)
    ctx.notes["execution_success"] = False

    plot_dir = algo.s3pogs_path / "results" / "KITTI_04" / "run_a" / "plot"
    plot_dir.mkdir(parents=True, exist_ok=True)
    (plot_dir / "trj_final.json").write_text("{}", encoding="utf-8")

    assert algo._find_raw_trajectory(request, ctx) is None


def test_s3pogs_find_raw_trajectory_ignores_pre_run_artifacts(tmp_path: Path) -> None:
    algo = S3POGSAlgorithm()
    algo.s3pogs_path = tmp_path / "S3PO-GS"
    request = _build_run_request(
        tmp_path,
        dataset_type="kitti",
        sequence_name="04",
        timestamps_by_frame={0: 0.0},
    )
    ctx = _build_context(request)
    ctx.notes["execution_success"] = True

    results_root = algo.s3pogs_path / "results" / "KITTI_04"
    old_trj = results_root / "old_run" / "plot" / "trj_final.json"
    new_trj = results_root / "new_run" / "plot" / "trj_final.json"
    old_trj.parent.mkdir(parents=True, exist_ok=True)
    new_trj.parent.mkdir(parents=True, exist_ok=True)
    old_trj.write_text("{}", encoding="utf-8")
    new_trj.write_text("{}", encoding="utf-8")

    now = time.time()
    ctx.notes["execution_started_at"] = now

    os.utime(old_trj, (now - 120.0, now - 120.0))
    os.utime(new_trj, (now + 2.0, now + 2.0))

    assert algo._find_raw_trajectory(request, ctx) == new_trj


def test_vggtslam_conversion_uses_request_timestamp_contract(tmp_path: Path) -> None:
    algo = VGGTSLAMAlgorithm()
    timestamp_ns = 1403636579763555584
    request = _build_run_request(
        tmp_path,
        dataset_type="euroc",
        sequence_name="V1_01_easy",
        timestamps_by_frame={0: timestamp_ns},
    )
    ctx = _build_context(request)

    raw_traj = tmp_path / "poses_raw.txt"
    raw_traj.write_text("0 1 2 3 0 0 0 1\n", encoding="utf-8")

    converted = algo._convert_raw_trajectory_to_tum(raw_traj, request, ctx)

    assert converted is not None
    line = converted.read_text(encoding="utf-8").strip()
    assert line.startswith(f"{timestamp_ns} ")


def test_vggtslam_conversion_accepts_timestamp_ids_from_raw_output(tmp_path: Path) -> None:
    algo = VGGTSLAMAlgorithm()
    timestamp_ns = 1403636579763555584
    request = _build_run_request(
        tmp_path,
        dataset_type="euroc",
        sequence_name="V1_01_easy",
        timestamps_by_frame={0: timestamp_ns},
    )
    ctx = _build_context(request)

    raw_traj = tmp_path / "poses_raw.txt"
    # VGGT-SLAM may emit dataset-native timestamp IDs in column 0.
    raw_traj.write_text(f"{timestamp_ns} 1 2 3 0 0 0 1\n", encoding="utf-8")

    converted = algo._convert_raw_trajectory_to_tum(raw_traj, request, ctx)

    assert converted is not None
    line = converted.read_text(encoding="utf-8").strip()
    assert line.startswith(f"{timestamp_ns} ")
