"""Tests for strict destination-aware metadata copying in pipeline root output."""

from pathlib import Path
from typing import List, Tuple

import pytest

from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import DatasetConfig, ExperimentConfig, OutputConfig
from slamadverseriallab.core.pipeline import Pipeline


class _DatasetWithDestMetadata:
    """Dataset stub exposing explicit destination metadata tuples."""

    def __init__(self, metadata_files: List[Tuple[Path, str, bool]]):
        self._metadata_files = metadata_files

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        return self._metadata_files


class _DatasetWithoutDestMetadata:
    """Dataset stub that does not implement destination-aware metadata."""

    def get_metadata_files_with_dest(self):
        raise NotImplementedError("missing get_metadata_files_with_dest")


def _make_pipeline(tmp_path: Path, max_frames: int = 2) -> Pipeline:
    config = Config(
        experiment=ExperimentConfig(name="metadata_copy_test"),
        dataset=DatasetConfig(type="kitti", path=str(tmp_path), max_frames=max_frames),
        perturbations=[],
        output=OutputConfig(
            base_dir=str(tmp_path / "results"),
            save_images=False,
            create_timestamp_dir=False,
        ),
    )
    pipeline = Pipeline(config)
    pipeline.output_dir = tmp_path / "output"
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    return pipeline


def test_root_copy_uses_destination_filenames_and_truncation(tmp_path: Path) -> None:
    trunc_src = tmp_path / "src_trunc.txt"
    trunc_src.write_text("# header\nline0\nline1\nline2\n", encoding="utf-8")

    full_src = tmp_path / "src_full.txt"
    full_src.write_text("calibration-data\n", encoding="utf-8")

    stub_dataset = _DatasetWithDestMetadata(
        metadata_files=[
            (trunc_src, "cam0_data.csv", True),
            (full_src, "cam0_sensor.yaml", False),
        ]
    )

    pipeline = _make_pipeline(tmp_path, max_frames=2)
    pipeline.dataset = stub_dataset
    pipeline._copy_dataset_files()

    copied_trunc = pipeline.output_dir / "cam0_data.csv"
    copied_full = pipeline.output_dir / "cam0_sensor.yaml"

    assert copied_trunc.exists()
    assert copied_full.exists()
    assert not (pipeline.output_dir / trunc_src.name).exists()
    assert not (pipeline.output_dir / full_src.name).exists()

    # Truncated file should keep comment + first 2 data lines.
    assert copied_trunc.read_text(encoding="utf-8").splitlines() == [
        "# header",
        "line0",
        "line1",
    ]
    assert copied_full.read_text(encoding="utf-8") == "calibration-data\n"


def test_root_copy_fails_on_duplicate_destination_filenames(tmp_path: Path) -> None:
    src_a = tmp_path / "cam0_data.csv"
    src_b = tmp_path / "cam1_data.csv"
    src_a.write_text("a\n", encoding="utf-8")
    src_b.write_text("b\n", encoding="utf-8")

    stub_dataset = _DatasetWithDestMetadata(
        metadata_files=[
            (src_a, "data.csv", True),
            (src_b, "data.csv", True),
        ]
    )

    pipeline = _make_pipeline(tmp_path, max_frames=2)
    pipeline.dataset = stub_dataset
    with pytest.raises(RuntimeError, match="Duplicate metadata destination filenames"):
        pipeline._copy_dataset_files()


def test_root_copy_requires_destination_metadata_api(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path, max_frames=2)
    pipeline.dataset = _DatasetWithoutDestMetadata()
    with pytest.raises(NotImplementedError, match="missing get_metadata_files_with_dest"):
        pipeline._copy_dataset_files()


def test_root_copy_requires_loaded_dataset(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path, max_frames=2)
    with pytest.raises(RuntimeError, match="Dataset must be loaded before copying dataset files"):
        pipeline._copy_dataset_files()
