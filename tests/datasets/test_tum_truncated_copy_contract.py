"""Tests for TUM truncated-copy metadata/image consistency."""

from pathlib import Path

import cv2
import numpy as np

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.tum import TUMDataset


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_tum_truncated_copy_links_files_referenced_by_associations(tmp_path: Path) -> None:
    source = tmp_path / "tum_source"
    source.mkdir(parents=True, exist_ok=True)

    # Older files that should NOT be selected when truncating by associations.
    _write_dummy_png(source / "rgb" / "1305031452.791720.png")
    _write_dummy_png(source / "rgb" / "1305031452.823674.png")

    # Files referenced by the first two association rows.
    _write_dummy_png(source / "rgb" / "1305031453.359684.png")
    _write_dummy_png(source / "rgb" / "1305031453.391690.png")
    _write_dummy_png(source / "depth" / "1305031453.374112.png")
    _write_dummy_png(source / "depth" / "1305031453.404816.png")

    (source / "associations.txt").write_text(
        "\n".join(
            [
                "1305031453.359684 rgb/1305031453.359684.png 1305031453.374112 depth/1305031453.374112.png",
                "1305031453.391690 rgb/1305031453.391690.png 1305031453.404816 depth/1305031453.404816.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Keep list files present to mirror real TUM directories.
    (source / "rgb.txt").write_text(
        "\n".join(
            [
                "1305031453.359684 rgb/1305031453.359684.png",
                "1305031453.391690 rgb/1305031453.391690.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (source / "depth.txt").write_text(
        "\n".join(
            [
                "1305031453.374112 depth/1305031453.374112.png",
                "1305031453.404816 depth/1305031453.404816.png",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = TUMDataset(DatasetConfig(type="tum", path=str(source), load_stereo=False))
    truncated = dataset.create_truncated_copy(2, output_dir=tmp_path / "tum_truncated")

    rgb_files = sorted(path.name for path in (truncated / "rgb").glob("*.png"))
    assert rgb_files == ["1305031453.359684.png", "1305031453.391690.png"]
    assert not (truncated / "rgb" / "1305031452.791720.png").exists()

    assoc_lines = [
        line.strip()
        for line in (truncated / "associations.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert len(assoc_lines) == 2

    for line in assoc_lines:
        parts = line.split()
        assert (truncated / parts[1]).exists()
        assert (truncated / parts[3]).exists()
