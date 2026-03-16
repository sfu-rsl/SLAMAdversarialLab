"""Tests ensuring dataset frame loading fails fast on unreadable RGB/stereo images."""

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


def _write_corrupt_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a valid image")


def test_kitti_load_frame_fails_fast_when_left_image_unreadable(tmp_path: Path) -> None:
    _write_corrupt_png(tmp_path / "image_2" / "000000.png")
    dataset = KittiDataset(DatasetConfig(type="kitti", path=str(tmp_path)))

    with pytest.raises(RuntimeError, match=r"KITTI frame 0: failed to load left image .*000000\.png"):
        dataset[0]


def test_tum_load_frame_fails_fast_when_rgb_image_unreadable(tmp_path: Path) -> None:
    _write_corrupt_png(tmp_path / "rgb" / "000000.png")
    (tmp_path / "associations.txt").write_text("0.0 rgb/000000.png\n", encoding="utf-8")
    dataset = TUMDataset(DatasetConfig(type="tum", path=str(tmp_path)))

    with pytest.raises(RuntimeError, match=r"TUM frame 0: failed to load RGB image .*000000\.png"):
        dataset[0]


def test_euroc_load_frame_fails_fast_when_left_image_unreadable(tmp_path: Path) -> None:
    _write_corrupt_png(tmp_path / "mav0" / "cam0" / "data" / "1400000000000000000.png")
    dataset = EuRoCDataset(DatasetConfig(type="euroc", path=str(tmp_path)))

    with pytest.raises(RuntimeError, match=r"EuRoC frame 0: failed to load left image .*1400000000000000000\.png"):
        dataset[0]


def test_euroc_stereo_load_frame_fails_fast_when_right_image_unreadable(tmp_path: Path) -> None:
    filename = "1400000000000000000.png"
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / filename)
    _write_corrupt_png(tmp_path / "mav0" / "cam1" / "data" / filename)
    dataset = EuRoCDataset(
        DatasetConfig(
            type="euroc",
            path=str(tmp_path),
            load_stereo=True,
        )
    )

    with pytest.raises(RuntimeError, match=r"EuRoC frame 0: failed to load right stereo image .*1400000000000000000\.png"):
        dataset[0]


def test_seven_scenes_load_frame_fails_fast_when_color_image_unreadable(tmp_path: Path) -> None:
    _write_corrupt_png(tmp_path / "seq-01" / "frame-000000.color.png")
    dataset = SevenScenesDataset(DatasetConfig(type="7scenes", path=str(tmp_path / "seq-01")))

    with pytest.raises(RuntimeError, match=r"7-Scenes frame 0: failed to load color image .*frame-000000\.color\.png"):
        dataset[0]
