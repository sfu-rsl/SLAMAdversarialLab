"""Tests for strict stereo-mode dataset requirements."""

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



def _create_minimal_tum_dataset(root: Path) -> TUMDataset:
    rgb_filename = "1305031102.175304.png"
    _write_dummy_png(root / "rgb" / rgb_filename)
    (root / "associations.txt").write_text(
        f"1305031102.175304 rgb/{rgb_filename}\n",
        encoding="utf-8",
    )
    return TUMDataset(
        DatasetConfig(
            type="tum",
            path=str(root),
            load_stereo=False,
        )
    )

def test_kitti_stereo_requires_image3_directory(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")

    config = DatasetConfig(
        type="kitti",
        path=str(tmp_path),
        load_stereo=True,
    )

    with pytest.raises(FileNotFoundError, match="Stereo mode requested but image_3/ directory not found"):
        KittiDataset(config)


def test_kitti_stereo_requires_right_frame_for_every_left_frame(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    _write_dummy_png(tmp_path / "image_2" / "000001.png")
    _write_dummy_png(tmp_path / "image_3" / "000000.png")

    config = DatasetConfig(
        type="kitti",
        path=str(tmp_path),
        load_stereo=True,
    )

    with pytest.raises(FileNotFoundError, match="requires right-camera image for every left frame"):
        KittiDataset(config)


def test_euroc_stereo_requires_cam1_directory(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / "1400000000000000000.png")

    config = DatasetConfig(
        type="euroc",
        path=str(tmp_path),
        load_stereo=True,
    )

    with pytest.raises(FileNotFoundError, match="Stereo mode requested but cam1/data/ directory not found"):
        EuRoCDataset(config)


def test_euroc_stereo_requires_right_frame_for_every_left_frame(tmp_path: Path) -> None:
    filename = "1400000000000000000.png"
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / filename)
    (tmp_path / "mav0" / "cam1" / "data").mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        type="euroc",
        path=str(tmp_path),
        load_stereo=True,
    )

    with pytest.raises(FileNotFoundError, match="requires right-camera image"):
        EuRoCDataset(config)


def test_kitti_create_pyslam_structure_fails_fast_when_stereo_right_dir_missing(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    _write_dummy_png(tmp_path / "image_3" / "000000.png")

    dataset = KittiDataset(
        DatasetConfig(
            type="kitti",
            path=str(tmp_path),
            load_stereo=True,
        )
    )

    perturbed_images = tmp_path / "perturbed"
    _write_dummy_png(perturbed_images / "image_2" / "000000.png")

    with pytest.raises(FileNotFoundError, match="active stereo mode requires right camera directory"):
        dataset.create_pyslam_structure(
            images_path=perturbed_images,
            temp_root=tmp_path / "pyslam_out",
            max_frames=1,
        )


def test_euroc_create_pyslam_structure_fails_fast_when_stereo_right_dir_missing(tmp_path: Path) -> None:
    filename = "1400000000000000000.png"
    _write_dummy_png(tmp_path / "mav0" / "cam0" / "data" / filename)
    _write_dummy_png(tmp_path / "mav0" / "cam1" / "data" / filename)

    dataset = EuRoCDataset(
        DatasetConfig(
            type="euroc",
            path=str(tmp_path),
            load_stereo=True,
        )
    )

    perturbed_images = tmp_path / "perturbed"
    _write_dummy_png(perturbed_images / "image_2" / filename)

    with pytest.raises(FileNotFoundError, match="active stereo mode requires right camera directory"):
        dataset.create_pyslam_structure(
            images_path=perturbed_images,
            temp_root=tmp_path / "pyslam_out",
            max_frames=1,
        )


def test_tum_rejects_right_camera_directory_request(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "rgb" / "1305031102.175304.png")
    (tmp_path / "associations.txt").write_text(
        "1305031102.175304 rgb/1305031102.175304.png\n",
        encoding="utf-8",
    )

    dataset = TUMDataset(
        DatasetConfig(
            type="tum",
            path=str(tmp_path),
            load_stereo=False,
        )
    )

    with pytest.raises(ValueError, match="monocular"):
        dataset.get_image_directory_name("right")


def test_tum_rejects_stereo_mode_request_in_config(tmp_path: Path) -> None:
    (tmp_path / "rgb").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="does not support stereo mode"):
        TUMDataset(
            DatasetConfig(
                type="tum",
                path=str(tmp_path),
                load_stereo=True,
            )
        )


def test_tum_requires_association_file_and_fails_fast_when_missing(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "rgb" / "1305031102.175304.png")

    with pytest.raises(RuntimeError, match="TUM datasets require an association file"):
        TUMDataset(
            DatasetConfig(
                type="tum",
                path=str(tmp_path),
                load_stereo=False,
            )
        )


def test_seven_scenes_rejects_right_camera_directory_request(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "seq-01" / "frame-000000.color.png")

    dataset = SevenScenesDataset(
        DatasetConfig(
            type="7scenes",
            path=str(tmp_path / "seq-01"),
            load_stereo=False,
        )
    )

    with pytest.raises(ValueError, match="single-camera"):
        dataset.get_image_directory_name("right")


def test_seven_scenes_rejects_stereo_mode_request_in_config(tmp_path: Path) -> None:
    (tmp_path / "seq-01").mkdir(parents=True, exist_ok=True)

    with pytest.raises(ValueError, match="does not support stereo mode"):
        SevenScenesDataset(
            DatasetConfig(
                type="7scenes",
                path=str(tmp_path / "seq-01"),
                load_stereo=True,
            )
        )


def test_tum_create_pyslam_structure_accepts_canonical_image2_input(tmp_path: Path) -> None:
    dataset = _create_minimal_tum_dataset(tmp_path / "tum_source")

    perturbed_images = tmp_path / "perturbed"
    _write_dummy_png(perturbed_images / "image_2" / "1305031102.175304.png")

    output_root = dataset.create_pyslam_structure(
        images_path=perturbed_images,
        temp_root=tmp_path / "pyslam_out",
        max_frames=1,
    )

    rgb_dir = output_root / dataset.sequence_name / "rgb"
    rgb_files = sorted(p.name for p in rgb_dir.glob("*.png"))
    assert rgb_files == ["1305031102.175304.png"]


def test_tum_create_pyslam_structure_prefers_rgb_over_canonical_when_both_exist(tmp_path: Path) -> None:
    dataset = _create_minimal_tum_dataset(tmp_path / "tum_source")

    perturbed_images = tmp_path / "perturbed"
    _write_dummy_png(perturbed_images / "rgb" / "from_rgb.png")
    _write_dummy_png(perturbed_images / "image_2" / "from_image2.png")

    output_root = dataset.create_pyslam_structure(
        images_path=perturbed_images,
        temp_root=tmp_path / "pyslam_out",
        max_frames=None,
    )

    rgb_dir = output_root / dataset.sequence_name / "rgb"
    rgb_files = sorted(p.name for p in rgb_dir.glob("*.png"))
    assert rgb_files == ["from_rgb.png"]


def test_tum_create_pyslam_structure_fails_when_no_image_source_exists(tmp_path: Path) -> None:
    dataset = _create_minimal_tum_dataset(tmp_path / "tum_source")

    perturbed_images = tmp_path / "perturbed"
    perturbed_images.mkdir(parents=True, exist_ok=True)
    (perturbed_images / "associations.txt").write_text("metadata only\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not resolve TUM RGB image directory"):
        dataset.create_pyslam_structure(
            images_path=perturbed_images,
            temp_root=tmp_path / "pyslam_out",
            max_frames=1,
        )
