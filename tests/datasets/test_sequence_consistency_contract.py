"""Tests for dataset sequence/path consistency contract."""

from pathlib import Path

import pytest

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.kitti import KittiDataset
from slamadverseriallab.datasets.tum import TUMDataset


def test_kitti_sequence_consistency_accepts_numeric_alias(tmp_path: Path) -> None:
    seq_dir = tmp_path / "04"
    image2_dir = seq_dir / "image_2"
    image2_dir.mkdir(parents=True, exist_ok=True)
    (image2_dir / "000000.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    dataset = KittiDataset(
        DatasetConfig(
            type="kitti",
            path=str(seq_dir),
            sequence="4",
            load_stereo=False,
        )
    )

    assert dataset.sequence_name == "4"
    assert dataset._frames[0]["sequence_id"] == "4"


def test_tum_sequence_consistency_accepts_rgbd_prefix_alias(tmp_path: Path) -> None:
    seq_dir = tmp_path / "rgbd_dataset_freiburg1_desk"
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "associations.txt").write_text(
        "0.0 rgb/000000.png 0.0 depth/000000.png\n",
        encoding="utf-8",
    )

    dataset = TUMDataset(
        DatasetConfig(
            type="tum",
            path=str(seq_dir),
            sequence="freiburg1_desk",
        )
    )

    assert dataset.sequence_name == "freiburg1_desk"
    assert dataset._frames[0]["sequence_id"] == "freiburg1_desk"


def test_tum_sequence_consistency_rejects_mismatch(tmp_path: Path) -> None:
    seq_dir = tmp_path / "rgbd_dataset_freiburg1_desk"
    seq_dir.mkdir(parents=True, exist_ok=True)
    (seq_dir / "associations.txt").write_text(
        "0.0 rgb/000000.png 0.0 depth/000000.png\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sequence mismatch"):
        TUMDataset(
            DatasetConfig(
                type="tum",
                path=str(seq_dir),
                sequence="freiburg2_desk",
            )
        )
