"""Tests for depth filename preservation in output writer."""

from pathlib import Path

import numpy as np

from slamadverseriallab.core.output import ImageWriter, OutputConfig, OutputFormat


def _create_writer(output_root: Path) -> ImageWriter:
    config = OutputConfig(
        format=OutputFormat.IMAGES,
        base_dir=output_root,
        save_images=False,
        save_depth=True,
        organize_by_module=False,
        organize_by_sequence=False,
    )
    writer = ImageWriter(config)
    writer.setup(experiment_name="")
    return writer


def test_image_writer_prefers_depth_filename_when_provided(tmp_path: Path) -> None:
    writer = _create_writer(tmp_path)
    depth = np.ones((4, 4), dtype=np.float32)

    writer.write_frame(
        {
            "depth": depth,
            "rgb_filename": "rgb_frame.png",
            "depth_filename": "depth_frame.png",
        },
        frame_idx=0,
    )

    assert (tmp_path / "depth" / "depth_frame.png").exists()
    assert not (tmp_path / "depth" / "rgb_frame.png").exists()


def test_image_writer_falls_back_to_rgb_filename_when_depth_filename_missing(tmp_path: Path) -> None:
    writer = _create_writer(tmp_path)
    depth = np.ones((4, 4), dtype=np.float32)

    writer.write_frame(
        {
            "depth": depth,
            "rgb_filename": "rgb_frame.png",
        },
        frame_idx=0,
    )

    assert (tmp_path / "depth" / "rgb_frame.png").exists()
