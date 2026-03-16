"""Tests for dataset camera intrinsics contract."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.euroc import EuRoCDataset
from slamadverseriallab.datasets.kitti import KittiDataset
from slamadverseriallab.datasets.seven_scenes import SevenScenesDataset
from slamadverseriallab.datasets.tum import TUMDataset


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def _write_kitti_calib(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "P2: 7.215377e+02 0.000000e+00 6.095593e+02 4.485728e+01 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03",
                "P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.395242e+02 0.000000e+00 7.215377e+02 1.728540e+02 2.163791e-01 0.000000e+00 0.000000e+00 1.000000e+00 2.745884e-03",
            ]
        ),
        encoding="utf-8",
    )


def _write_euroc_sensor_yaml(path: Path, fx: float, fy: float, cx: float, cy: float, tx: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "sensor_type: camera",
                f"intrinsics: [{fx}, {fy}, {cx}, {cy}]",
                "resolution: [752, 480]",
                "T_BS:",
                "  rows: 4",
                "  cols: 4",
                f"  data: [1.0, 0.0, 0.0, {tx}, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]",
            ]
        ),
        encoding="utf-8",
    )


def test_kitti_camera_intrinsics_left_and_right(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    _write_dummy_png(tmp_path / "image_3" / "000000.png")
    _write_kitti_calib(tmp_path / "calib.txt")

    dataset = KittiDataset(
        DatasetConfig(type="kitti", path=str(tmp_path), load_stereo=True)
    )

    left = dataset.get_camera_intrinsics("left")
    right = dataset.get_camera_intrinsics("right")

    assert left is not None
    assert right is not None

    for intrinsics in (left, right):
        assert intrinsics.fx > 0
        assert intrinsics.fy > 0
        assert intrinsics.cx > 0
        assert intrinsics.cy > 0

    assert left.baseline is not None
    assert left.baseline > 0


def test_kitti_camera_intrinsics_missing_calibration_returns_none(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")

    dataset = KittiDataset(
        DatasetConfig(type="kitti", path=str(tmp_path), load_stereo=False)
    )

    assert dataset.get_camera_intrinsics("left") is None


def test_euroc_camera_intrinsics_left_and_right(tmp_path: Path) -> None:
    filename = "1400000000000000000.png"
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / filename)
    _write_dummy_png(tmp_path / "mav0" / "cam1" / "data" / filename)

    _write_euroc_sensor_yaml(
        tmp_path / "mav0" / "cam0" / "sensor.yaml",
        fx=458.654,
        fy=457.296,
        cx=367.215,
        cy=248.375,
        tx=0.0,
    )
    _write_euroc_sensor_yaml(
        tmp_path / "mav0" / "cam1" / "sensor.yaml",
        fx=457.587,
        fy=456.134,
        cx=379.999,
        cy=255.238,
        tx=0.110073808127187,
    )

    dataset = EuRoCDataset(
        DatasetConfig(type="euroc", path=str(tmp_path), load_stereo=True)
    )

    left = dataset.get_camera_intrinsics("left")
    right = dataset.get_camera_intrinsics("right")

    assert left is not None
    assert right is not None

    for intrinsics in (left, right):
        assert intrinsics.fx > 0
        assert intrinsics.fy > 0
        assert intrinsics.cx > 0
        assert intrinsics.cy > 0
        assert intrinsics.width is not None and intrinsics.width > 0
        assert intrinsics.height is not None and intrinsics.height > 0

    assert left.baseline is not None
    assert left.baseline > 0
    assert left.baseline == pytest.approx(0.110073808127187, rel=1e-3)


def test_euroc_camera_intrinsics_missing_sensor_yaml_returns_none(tmp_path: Path) -> None:
    filename = "1400000000000000000.png"
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / filename)

    dataset = EuRoCDataset(
        DatasetConfig(type="euroc", path=str(tmp_path), load_stereo=False)
    )

    assert dataset.get_camera_intrinsics("left") is None


def test_tum_camera_intrinsics_sequence_specific(tmp_path: Path) -> None:
    tum_path = tmp_path / "rgbd_dataset_freiburg2_desk"
    image_name = "1305031102.175304.png"

    _write_dummy_png(tum_path / "rgb" / image_name)
    (tum_path / "associations.txt").write_text(
        f"1305031102.175304 rgb/{image_name}\n",
        encoding="utf-8",
    )

    dataset = TUMDataset(
        DatasetConfig(type="tum", path=str(tum_path), load_stereo=False)
    )

    intrinsics = dataset.get_camera_intrinsics("left")
    assert intrinsics is not None
    assert intrinsics.fx == pytest.approx(520.9)
    assert intrinsics.fy == pytest.approx(521.0)
    assert intrinsics.cx == pytest.approx(325.1)
    assert intrinsics.cy == pytest.approx(249.7)


def test_tum_camera_intrinsics_right_camera_unavailable(tmp_path: Path) -> None:
    image_name = "1305031102.175304.png"
    _write_dummy_png(tmp_path / "rgb" / image_name)
    (tmp_path / "associations.txt").write_text(
        f"1305031102.175304 rgb/{image_name}\n",
        encoding="utf-8",
    )

    dataset = TUMDataset(
        DatasetConfig(type="tum", path=str(tmp_path), load_stereo=False)
    )

    assert dataset.get_camera_intrinsics("right") is None


def test_seven_scenes_camera_intrinsics_right_camera_unavailable(tmp_path: Path) -> None:
    sequence_dir = tmp_path / "seq-01"
    _write_dummy_png(sequence_dir / "frame-000000.color.png")

    dataset = SevenScenesDataset(
        DatasetConfig(type="7scenes", path=str(sequence_dir), load_stereo=False)
    )

    left = dataset.get_camera_intrinsics("left")
    assert left is not None
    assert left.fx > 0
    assert left.fy > 0
    assert dataset.get_camera_intrinsics("right") is None
