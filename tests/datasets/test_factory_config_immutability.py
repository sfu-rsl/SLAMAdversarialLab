"""Tests ensuring dataset factory does not mutate caller config."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.base import Dataset
from slamadverseriallab.datasets.factory import _registry, create_dataset


class _ResolvePathDataset(Dataset):
    """Minimal dataset used to validate factory behavior."""

    _resolved_path: Path

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        return str(cls._resolved_path)

    def _validate_path(self) -> None:
        # Test dataset does not require real path validation.
        return

    def _load_dataset(self) -> None:
        self._frames = [
            {
                "image_path": "dummy.png",
                "timestamp": 0.0,
                "sequence_id": "seq",
                "frame_id": 0,
            }
        ]

    def _load_frame(self, idx: int) -> Dict[str, Any]:
        frame_info = self._frames[idx]
        return {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": None,
            "timestamp": frame_info["timestamp"],
            "sequence_id": frame_info["sequence_id"],
            "frame_id": frame_info["frame_id"],
        }

    def create_truncated_copy(self, max_frames: int, output_dir: Path | None = None) -> Path:
        return Path(output_dir) if output_dir is not None else self.path

    def get_ground_truth_path(self) -> Path | None:
        return None

    def get_metadata_files_with_dest(self) -> List:
        return []

    def get_image_directory_name(self, camera: str = "left") -> str:
        return "image_2"


def test_create_dataset_does_not_mutate_input_config_when_resolving_sequence(tmp_path: Path) -> None:
    dataset_name = "factory_immutability_test"
    alias_name = "factory_immutability_test_alias"
    resolved_path = tmp_path / "resolved_dataset"
    resolved_path.mkdir(parents=True, exist_ok=True)

    _ResolvePathDataset._resolved_path = resolved_path
    _registry.register(dataset_name, _ResolvePathDataset, aliases=[alias_name])

    config = DatasetConfig(
        type=dataset_name,
        sequence="resolved_dataset",
        path=None,
    )

    try:
        dataset = create_dataset(config)
        assert config.path is None
        assert dataset.path == resolved_path
        assert dataset.config.path == str(resolved_path)
    finally:
        _registry._datasets.pop(dataset_name, None)
        _registry._metadata.pop(dataset_name, None)
        _registry._aliases.pop(alias_name, None)


def test_create_dataset_fails_when_config_sequence_and_path_sequence_mismatch(tmp_path: Path) -> None:
    dataset_name = "factory_sequence_mismatch_test"
    resolved_path = tmp_path / "actual_sequence"
    resolved_path.mkdir(parents=True, exist_ok=True)

    _ResolvePathDataset._resolved_path = resolved_path
    _registry.register(dataset_name, _ResolvePathDataset)

    config = DatasetConfig(
        type=dataset_name,
        sequence="different_sequence",
        path=None,
    )

    try:
        with pytest.raises(RuntimeError, match="sequence mismatch"):
            create_dataset(config)
    finally:
        _registry._datasets.pop(dataset_name, None)
        _registry._metadata.pop(dataset_name, None)
