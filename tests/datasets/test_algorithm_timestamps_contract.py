"""Tests for dataset-provided algorithm timestamp mappings."""

from pathlib import Path

import cv2
import numpy as np

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.euroc import EuRoCDataset
from slamadverseriallab.datasets.kitti import KittiDataset


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_kitti_get_algorithm_timestamps_reads_times_txt(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    _write_dummy_png(tmp_path / "image_2" / "000001.png")
    (tmp_path / "times.txt").write_text("0.000000\n0.100000\n", encoding="utf-8")

    dataset = KittiDataset(DatasetConfig(type="kitti", path=str(tmp_path), load_stereo=False))

    assert dataset.get_algorithm_timestamps() == {0: 0.0, 1: 0.1}


def test_euroc_get_algorithm_timestamps_uses_nanoseconds(tmp_path: Path) -> None:
    ts0 = 1403636579763555584
    ts1 = 1403636579813555584
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / f"{ts0}.png")
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / f"{ts1}.png")
    (tmp_path / "mav0" / "cam0" / "data.csv").write_text(
        "\n".join(
            [
                "#timestamp [ns],filename",
                f"{ts0},{ts0}.png",
                f"{ts1},{ts1}.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = EuRoCDataset(DatasetConfig(type="euroc", path=str(tmp_path), load_stereo=False))

    assert dataset.get_algorithm_timestamps() == {0: ts0, 1: ts1}
