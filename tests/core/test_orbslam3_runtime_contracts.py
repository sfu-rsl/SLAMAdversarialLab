"""Tests for ORB-SLAM3 runtime camera/timestamp contract enforcement."""

from pathlib import Path

import pytest

from slamadverseriallab.algorithms.orbslam3 import ORBSLAM3Algorithm
from slamadverseriallab.algorithms.types import SLAMRunRequest, SLAMRuntimeContext, SensorMode


def _build_request(
    tmp_path: Path,
    *,
    dataset_type: str,
    sensor_mode: SensorMode,
    extras: dict,
) -> SLAMRunRequest:
    dataset_path = tmp_path / "dataset"
    output_dir = tmp_path / "output"
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return SLAMRunRequest(
        dataset_path=dataset_path,
        slam_config="dummy_config",
        output_dir=output_dir,
        dataset_type=dataset_type,
        sensor_mode=sensor_mode,
        sequence_name="V1_01_easy" if dataset_type == "euroc" else "04",
        extras=extras,
    )


def _build_context(request: SLAMRunRequest) -> SLAMRuntimeContext:
    return SLAMRuntimeContext(
        request=request,
        config_is_external=False,
        resolved_config_path=None,
        internal_config_name="EuRoC.yaml" if request.dataset_type == "euroc" else "KITTI04-12.yaml",
        sequence_name=request.sequence_name,
    )


def test_orbslam3_kitti_stage_uses_mount_contract_without_source_mutation(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="kitti",
        sensor_mode=SensorMode.MONO,
        extras={"camera_dirs": {"left": "left_cam"}, "camera_paths": {}},
    )
    ctx = _build_context(request)
    left_cam_dir = request.dataset_path / "left_cam"
    left_cam_dir.mkdir(parents=True, exist_ok=True)
    (left_cam_dir / "000000.png").write_bytes(b"x")
    (request.dataset_path / "calib.txt").write_text("P0: 1 0 0 0\n", encoding="utf-8")
    (request.dataset_path / "times.txt").write_text("0.0\n", encoding="utf-8")
    request.extras["camera_paths"] = {"left": str(left_cam_dir.resolve())}

    algo = ORBSLAM3Algorithm()
    staged = algo._stage_dataset(request, ctx)

    assert staged is not None
    assert staged != request.dataset_path
    assert not (request.dataset_path / "image_0").exists()
    assert (staged / "image_0").is_dir()
    assert (staged / "calib.txt").exists()
    assert (staged / "times.txt").exists()
    assert ctx.staging_artifacts.get("kitti_temp_dir") == staged
    assert ctx.staging_artifacts.get("kitti_image_mounts") == [
        (str(left_cam_dir.resolve()), "/dataset/image_0")
    ]

    algo._cleanup_staged_dataset(request, ctx)
    assert not staged.exists()


def test_orbslam3_kitti_stereo_stage_fails_without_right_camera_dir(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="kitti",
        sensor_mode=SensorMode.STEREO,
        extras={"camera_dirs": {"left": "left_cam"}},
    )
    ctx = _build_context(request)
    left_cam_dir = request.dataset_path / "left_cam"
    left_cam_dir.mkdir(parents=True, exist_ok=True)

    algo = ORBSLAM3Algorithm()
    with pytest.raises(RuntimeError, match="camera_dirs mapping for role 'right'"):
        algo._stage_dataset(request, ctx)


def test_orbslam3_preflight_tum_requires_existing_association_even_with_rgb_depth_lists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="tum",
        sensor_mode=SensorMode.RGBD,
        extras={},
    )
    ctx = _build_context(request)

    (request.dataset_path / "rgb.txt").write_text("0.0 rgb/0.png\n", encoding="utf-8")
    (request.dataset_path / "depth.txt").write_text("0.0 depth/0.png\n", encoding="utf-8")

    class _Result:
        returncode = 0

    monkeypatch.setattr("slamadverseriallab.algorithms.orbslam3.shutil.which", lambda _bin: "/usr/bin/docker")
    monkeypatch.setattr(
        "slamadverseriallab.algorithms.orbslam3.subprocess.run",
        lambda *_args, **_kwargs: _Result(),
    )

    algo = ORBSLAM3Algorithm()
    with pytest.raises(RuntimeError, match="requires an existing TUM association file"):
        algo._preflight_checks(request, ctx)


@pytest.mark.parametrize(
    "assoc_name",
    ["associations.txt", "association.txt", "associate.txt", "assoc.txt"],
)
def test_orbslam3_tum_stage_accepts_existing_association_aliases(
    tmp_path: Path,
    assoc_name: str,
) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="tum",
        sensor_mode=SensorMode.RGBD,
        extras={},
    )
    ctx = _build_context(request)
    (request.dataset_path / assoc_name).write_text("0.0 rgb/0.png 0.0 depth/0.png\n", encoding="utf-8")

    algo = ORBSLAM3Algorithm()
    staged = algo._stage_dataset(request, ctx)

    assert staged == request.dataset_path
    assert ctx.staging_artifacts.get("association_file") == assoc_name


def test_orbslam3_tum_stage_resolver_is_called_in_strict_mode(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="tum",
        sensor_mode=SensorMode.RGBD,
        extras={},
    )
    ctx = _build_context(request)
    expected_assoc = request.dataset_path / "associations.txt"
    expected_assoc.write_text("0.0 rgb/0.png 0.0 depth/0.png\n", encoding="utf-8")
    captured = {}

    def _fake_resolver(dataset_path: Path, generate_if_missing: bool, log):
        captured["dataset_path"] = dataset_path
        captured["generate_if_missing"] = generate_if_missing
        captured["log"] = log
        return expected_assoc

    monkeypatch.setattr(
        "slamadverseriallab.algorithms.orbslam3.resolve_tum_association_for_orbslam3",
        _fake_resolver,
    )

    algo = ORBSLAM3Algorithm()
    staged = algo._stage_dataset(request, ctx)

    assert staged == request.dataset_path
    assert captured["dataset_path"] == request.dataset_path
    assert captured["generate_if_missing"] is False
    assert captured["log"] is not None
    assert ctx.staging_artifacts.get("association_file") == "associations.txt"


def test_orbslam3_euroc_stage_uses_camera_paths_and_timestamps_contract(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        extras={
            "camera_dirs": {"left": "mav0/cam0/data", "right": "mav0/cam1/data"},
            "camera_paths": {},
            "timestamps_by_frame": {0: 1403715273262142976, 1: 1403715273312142976},
        },
    )
    ctx = _build_context(request)

    left_dir = request.dataset_path / "custom_left"
    right_dir = request.dataset_path / "custom_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    (left_dir / "1403715273262142976.png").write_bytes(b"x")
    (left_dir / "1403715273312142976.png").write_bytes(b"x")
    (right_dir / "1403715273262142976.png").write_bytes(b"x")
    (right_dir / "1403715273312142976.png").write_bytes(b"x")
    (request.dataset_path / "cam0_sensor.yaml").write_text("dummy: true\n", encoding="utf-8")
    (request.dataset_path / "cam1_sensor.yaml").write_text("dummy: true\n", encoding="utf-8")

    request.extras["camera_paths"] = {
        "left": str(left_dir.resolve()),
        "right": str(right_dir.resolve()),
    }

    algo = ORBSLAM3Algorithm()
    staged = algo._stage_dataset(request, ctx)

    assert staged is not None
    assert (staged / "orbslam3_timestamps.txt").read_text(encoding="utf-8").splitlines() == [
        "1403715273262142976",
        "1403715273312142976",
    ]
    cam0_csv = (staged / "mav0" / "cam0" / "data.csv").read_text(encoding="utf-8").splitlines()
    cam1_csv = (staged / "mav0" / "cam1" / "data.csv").read_text(encoding="utf-8").splitlines()
    assert cam0_csv[0] == "#timestamp [ns],filename"
    assert cam1_csv[0] == "#timestamp [ns],filename"
    assert len(cam0_csv) == 3
    assert len(cam1_csv) == 3
    assert (staged / "mav0" / "cam0" / "sensor.yaml").exists()
    assert (staged / "mav0" / "cam1" / "sensor.yaml").exists()

    mounts = ctx.staging_artifacts.get("euroc_image_mounts")
    assert mounts == [
        (str(left_dir.resolve()), "/dataset/mav0/cam0/data"),
        (str(right_dir.resolve()), "/dataset/mav0/cam1/data"),
    ]

    algo._cleanup_staged_dataset(request, ctx)
    assert not staged.exists()


def test_orbslam3_euroc_stage_fails_when_timestamp_count_mismatches_images(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        extras={
            "camera_dirs": {"left": "mav0/cam0/data", "right": "mav0/cam1/data"},
            "camera_paths": {},
            "timestamps_by_frame": {0: 1403715273262142976},
        },
    )
    ctx = _build_context(request)

    left_dir = request.dataset_path / "custom_left"
    right_dir = request.dataset_path / "custom_right"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    (left_dir / "1403715273262142976.png").write_bytes(b"x")
    (left_dir / "1403715273312142976.png").write_bytes(b"x")
    (right_dir / "1403715273262142976.png").write_bytes(b"x")
    (right_dir / "1403715273312142976.png").write_bytes(b"x")
    request.extras["camera_paths"] = {
        "left": str(left_dir.resolve()),
        "right": str(right_dir.resolve()),
    }

    algo = ORBSLAM3Algorithm()
    with pytest.raises(RuntimeError, match="timestamp mapping does not match image count"):
        algo._stage_dataset(request, ctx)


def test_orbslam3_preflight_requires_camera_paths_for_euroc(tmp_path: Path, monkeypatch) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        extras={
            "camera_dirs": {"left": "mav0/cam0/data", "right": "mav0/cam1/data"},
            "timestamps_by_frame": {0: 1403715273262142976},
        },
    )
    ctx = _build_context(request)

    class _Result:
        returncode = 0

    monkeypatch.setattr("slamadverseriallab.algorithms.orbslam3.shutil.which", lambda _bin: "/usr/bin/docker")
    monkeypatch.setattr(
        "slamadverseriallab.algorithms.orbslam3.subprocess.run",
        lambda *_args, **_kwargs: _Result(),
    )

    algo = ORBSLAM3Algorithm()
    with pytest.raises(RuntimeError, match="camera_paths"):
        algo._preflight_checks(request, ctx)


def test_orbslam3_execution_spec_uses_staged_euroc_timestamps_file(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="euroc",
        sensor_mode=SensorMode.STEREO,
        extras={},
    )
    ctx = _build_context(request)
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir,
        "dataset_type": "euroc",
        "is_stereo": True,
        "sequence_name": "V1_01_easy",
        "is_external": False,
        "kitti_image_mounts": None,
        "euroc_image_mounts": [],
    }

    algo = ORBSLAM3Algorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    cmd_text = " ".join(spec.cmd)
    assert "/dataset/orbslam3_timestamps.txt" in cmd_text
    assert "EuRoC_TimeStamps" not in cmd_text


def test_orbslam3_execution_spec_tum_requires_staged_association_without_fallback_lookup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="tum",
        sensor_mode=SensorMode.RGBD,
        extras={},
    )
    ctx = _build_context(request)
    ctx.execution_inputs = {
        "dataset_path": request.dataset_path,
        "output_dir": request.output_dir,
        "dataset_type": "tum",
        "is_stereo": False,
        "sequence_name": request.sequence_name,
        "is_external": False,
        "kitti_image_mounts": None,
        "euroc_image_mounts": None,
    }

    algo = ORBSLAM3Algorithm()

    def _unexpected_fallback(*_args, **_kwargs):
        raise AssertionError("build spec must not perform TUM association fallback lookup")

    monkeypatch.setattr(algo, "_find_association_file", _unexpected_fallback)

    spec = algo._build_execution_spec(request, ctx)
    assert spec is None


def test_orbslam3_execution_spec_kitti_mounts_camera_paths_to_image0_and_image1(tmp_path: Path) -> None:
    request = _build_request(
        tmp_path,
        dataset_type="kitti",
        sensor_mode=SensorMode.STEREO,
        extras={},
    )
    ctx = _build_context(request)

    staged_root = tmp_path / "staged_kitti"
    staged_root.mkdir(parents=True, exist_ok=True)
    left_dir = tmp_path / "left_contract"
    right_dir = tmp_path / "right_contract"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    ctx.execution_inputs = {
        "dataset_path": staged_root,
        "output_dir": request.output_dir,
        "dataset_type": "kitti",
        "is_stereo": True,
        "sequence_name": request.sequence_name,
        "is_external": False,
        "kitti_image_mounts": [
            (str(left_dir.resolve()), "/dataset/image_0"),
            (str(right_dir.resolve()), "/dataset/image_1"),
        ],
        "euroc_image_mounts": None,
    }

    algo = ORBSLAM3Algorithm()
    spec = algo._build_execution_spec(request, ctx)

    assert spec is not None
    cmd_text = " ".join(spec.cmd)
    assert f"{left_dir.resolve()}:/dataset/image_0:ro" in cmd_text
    assert f"{right_dir.resolve()}:/dataset/image_1:ro" in cmd_text
