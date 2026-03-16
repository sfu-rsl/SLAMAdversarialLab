"""Tests for MASt3R sequence-name handling."""

import os
from pathlib import Path

import pytest

from slamadverseriallab.algorithms.mast3rslam import MASt3RSLAMAlgorithm
from slamadverseriallab.algorithms.types import SLAMRunRequest, SensorMode


class _DummyProcess:
    returncode = 0


def _build_request(
    *,
    dataset_path: Path,
    left_camera_path: str | None,
    timestamps_by_frame: dict[int, float | int] | None,
) -> SLAMRunRequest:
    output_dir = dataset_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    extras = {}
    if left_camera_path is not None:
        extras["camera_paths"] = {"left": left_camera_path}
    if timestamps_by_frame is not None:
        extras["timestamps_by_frame"] = timestamps_by_frame

    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="base",
        output_dir=output_dir,
        dataset_type="tum",
        sensor_mode=SensorMode.MONO,
        sequence_name="freiburg2_desk",
        extras=extras,
    )


def test_mast3r_prepare_dataset_stages_tum_root_from_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    image_dir = dataset_path / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "1305031102.175304.png").write_bytes(b"png")
    (image_dir / "1305031102.207344.png").write_bytes(b"png")

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=str(image_dir),
        timestamps_by_frame={
            0: 1305031102.175304,
            1: 1305031102.207344,
        },
    )

    algorithm = MASt3RSLAMAlgorithm()

    prepared = algorithm._prepare_dataset(
        request=request,
        sequence_name="freiburg2_desk",
    )

    assert prepared.name == "rgbd_dataset_freiburg2_freiburg2_desk"
    assert prepared.is_dir()
    assert "tum" in prepared.parts

    rgb_link = prepared / "rgb"
    assert rgb_link.is_symlink()
    assert rgb_link.resolve() == image_dir.resolve()

    rgb_lines = (prepared / "rgb.txt").read_text(encoding="utf-8").splitlines()
    assert rgb_lines == [
        "1305031102.175304 rgb/1305031102.175304.png",
        "1305031102.207344 rgb/1305031102.207344.png",
    ]

    algorithm._cleanup_temp_tum_link()


def test_mast3r_prepare_dataset_fails_without_left_camera_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    dataset_path.mkdir(parents=True, exist_ok=True)

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=None,
        timestamps_by_frame={0: 1.0},
    )

    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="camera_paths"):
        algorithm._prepare_dataset(request=request, sequence_name="freiburg2_desk")


def test_mast3r_prepare_dataset_fails_without_timestamps_contract(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    image_dir = dataset_path / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "1305031102.175304.png").write_bytes(b"png")

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=str(image_dir),
        timestamps_by_frame=None,
    )

    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="timestamps_by_frame"):
        algorithm._prepare_dataset(request=request, sequence_name="freiburg2_desk")


def test_mast3r_prepare_dataset_fails_with_non_contiguous_timestamp_indices(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    image_dir = dataset_path / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "1305031102.175304.png").write_bytes(b"png")
    (image_dir / "1305031102.207344.png").write_bytes(b"png")

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=str(image_dir),
        timestamps_by_frame={
            0: 1305031102.175304,
            2: 1305031102.207344,
        },
    )

    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="contiguous frame indices"):
        algorithm._prepare_dataset(request=request, sequence_name="freiburg2_desk")


def test_mast3r_prepare_dataset_fails_with_non_increasing_timestamps(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    image_dir = dataset_path / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "1305031102.175304.png").write_bytes(b"png")
    (image_dir / "1305031102.207344.png").write_bytes(b"png")

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=str(image_dir),
        timestamps_by_frame={
            0: 1305031102.207344,
            1: 1305031102.175304,
        },
    )

    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="strictly increasing"):
        algorithm._prepare_dataset(request=request, sequence_name="freiburg2_desk")


def test_mast3r_prepare_dataset_fails_when_timestamp_image_counts_mismatch(tmp_path: Path) -> None:
    dataset_path = tmp_path / "perturbed" / "fog"
    image_dir = dataset_path / "image_2"
    image_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / "1305031102.175304.png").write_bytes(b"png")
    (image_dir / "1305031102.207344.png").write_bytes(b"png")

    request = _build_request(
        dataset_path=dataset_path,
        left_camera_path=str(image_dir),
        timestamps_by_frame={0: 1305031102.175304},
    )

    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="counts to match"):
        algorithm._prepare_dataset(request=request, sequence_name="freiburg2_desk")


@pytest.mark.parametrize(
    ("sequence_name", "expected"),
    [
        ("freiburg1_desk", "freiburg1"),
        ("freiburg1_xyz", "freiburg1"),
        ("rgbd_dataset_freiburg1_desk", "freiburg1"),
        ("fr1_desk", "freiburg1"),
        ("fr2_large_loop", "freiburg2"),
        ("freiburg3_structure_texture_far", "freiburg3"),
    ],
)
def test_mast3r_resolve_freiburg_id_accepts_valid_sequence_markers(
    sequence_name: str, expected: str
) -> None:
    algorithm = MASt3RSLAMAlgorithm()
    assert algorithm._resolve_freiburg_id_from_sequence(sequence_name) == expected


@pytest.mark.parametrize(
    "sequence_name",
    [
        "",
        "desk",
        "rgbd_dataset_room1",
    ],
)
def test_mast3r_resolve_freiburg_id_fails_when_marker_missing(sequence_name: str) -> None:
    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="could not resolve Freiburg variant|requires dataset.sequence"):
        algorithm._resolve_freiburg_id_from_sequence(sequence_name)


def test_mast3r_resolve_freiburg_id_fails_when_marker_ambiguous() -> None:
    algorithm = MASt3RSLAMAlgorithm()
    with pytest.raises(ValueError, match="ambiguous Freiburg markers"):
        algorithm._resolve_freiburg_id_from_sequence("freiburg1_freiburg2_mixed")


def test_mast3r_run_prefers_sequence_named_trajectory_over_latest(tmp_path: Path, monkeypatch) -> None:
    mast3r_path = tmp_path / "mast3r"
    logs_dir = mast3r_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    expected_traj = logs_dir / "rgbd_dataset_freiburg3_fr3_desk.txt"
    expected_traj.write_text("expected", encoding="utf-8")

    unrelated_traj = logs_dir / "unrelated_latest.txt"
    unrelated_traj.write_text("wrong", encoding="utf-8")

    # Make unrelated file newer so a naive "pick latest" fallback would choose it.
    now = max(expected_traj.stat().st_mtime, unrelated_traj.stat().st_mtime) + 10
    os.utime(unrelated_traj, (now, now))

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "base.yaml"
    config_file.write_text("dummy: true\n", encoding="utf-8")

    algorithm = MASt3RSLAMAlgorithm()
    algorithm.mast3r_path = mast3r_path

    dummy_process = _DummyProcess()
    monkeypatch.setattr(algorithm, "_spawn_streaming_process", lambda _cmd: dummy_process)
    monkeypatch.setattr(algorithm, "_stream_process_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(algorithm, "_wait_for_process", lambda _process, timeout_seconds: None)

    success = algorithm._run_mast3rslam(
        dataset_path=tmp_path / "path_name_should_not_matter",
        config_file=config_file,
        output_dir=output_dir,
        log_basenames=["rgbd_dataset_freiburg3_fr3_desk", "fr3_desk"],
    )

    assert success is True
    assert (output_dir / "CameraTrajectory.txt").read_text(encoding="utf-8") == "expected"


def test_mast3r_run_fails_when_expected_trajectory_log_missing(tmp_path: Path, monkeypatch) -> None:
    mast3r_path = tmp_path / "mast3r"
    logs_dir = mast3r_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Unrelated logs exist, but expected sequence logs do not.
    (logs_dir / "unrelated_latest.txt").write_text("wrong", encoding="utf-8")
    (logs_dir / "another_run.txt").write_text("wrong2", encoding="utf-8")

    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    config_file = tmp_path / "base.yaml"
    config_file.write_text("dummy: true\n", encoding="utf-8")

    algorithm = MASt3RSLAMAlgorithm()
    algorithm.mast3r_path = mast3r_path

    dummy_process = _DummyProcess()
    monkeypatch.setattr(algorithm, "_spawn_streaming_process", lambda _cmd: dummy_process)
    monkeypatch.setattr(algorithm, "_stream_process_output", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(algorithm, "_wait_for_process", lambda _process, timeout_seconds: None)

    success = algorithm._run_mast3rslam(
        dataset_path=tmp_path / "path_name_should_not_matter",
        config_file=config_file,
        output_dir=output_dir,
        log_basenames=["rgbd_dataset_freiburg3_fr3_desk", "fr3_desk"],
    )

    assert success is False
    assert not (output_dir / "CameraTrajectory.txt").exists()
