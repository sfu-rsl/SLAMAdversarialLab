"""Regression tests for canonical dataset frame metadata schema."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytest

from slamadverseriallab.config.schema import DatasetConfig
from slamadverseriallab.datasets.base import Dataset, MockDataset


class _StaticDataset(Dataset):
    """Minimal dataset implementation for schema/metadata unit tests."""

    def __init__(
        self,
        config: DatasetConfig,
        frames: List[Dict[str, Any]],
        is_stereo: bool = False,
        native_depth_dir: Optional[Path] = None,
        metadata_files_with_dest: Optional[List[tuple]] = None,
    ) -> None:
        self._source_frames = frames
        self._is_stereo = is_stereo
        self._native_depth_dir = native_depth_dir
        self._metadata_files_with_dest = metadata_files_with_dest or []
        self.native_decode_calls = 0
        super().__init__(config)

    @property
    def is_stereo(self) -> bool:
        return self._is_stereo

    @property
    def supports_stereo(self) -> bool:
        return self._is_stereo

    def _validate_path(self) -> None:
        # Test datasets do not need on-disk path validation.
        return

    def _load_dataset(self) -> None:
        self._frames = list(self._source_frames)

    def _load_frame(self, idx: int) -> Dict[str, Any]:
        frame_info = self._frames[idx]
        return {
            "image": np.zeros((2, 2, 3), dtype=np.uint8),
            "depth": None,
            "timestamp": frame_info["timestamp"],
            "sequence_id": frame_info["sequence_id"],
            "frame_id": frame_info["frame_id"],
            "rgb_filename": Path(frame_info["image_path"]).name,
        }

    def get_depth_directory_path(self) -> Optional[Path]:
        return self._native_depth_dir

    def get_metadata_files_with_dest(self):
        return self._metadata_files_with_dest

    def decode_native_depth(self, depth: np.ndarray) -> np.ndarray:
        self.native_decode_calls += 1
        return depth.astype(np.float32) / 1000.0

    def get_image_directory_name(self, camera: str = "left") -> str:
        if camera == "right":
            if not self._is_stereo:
                raise ValueError("right camera unavailable")
            return "rgb_right"
        return "rgb"


def test_mock_dataset_uses_canonical_required_fields() -> None:
    dataset = MockDataset(
        DatasetConfig(type="mock", path="mock", max_frames=3),
        num_frames=3,
    )

    required_fields = set(Dataset.REQUIRED_FRAME_FIELDS.keys())
    assert len(dataset) == 3
    assert len(dataset.get_image_paths("left")) == 3
    for frame in dataset._frames:
        assert required_fields.issubset(frame.keys())


def test_validate_frame_schema_rejects_missing_required_field(tmp_path: Path) -> None:
    frames = [
        {
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))

    with pytest.raises(ValueError, match="missing required field 'image_path'"):
        _StaticDataset(config=config, frames=frames)


def test_validate_frame_schema_rejects_stereo_without_right_camera(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))

    with pytest.raises(ValueError, match="missing 'image_path_right'"):
        _StaticDataset(config=config, frames=frames, is_stereo=True)


def test_capability_contract_rejects_stereo_request_for_mono_dataset(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path), load_stereo=True)

    with pytest.raises(ValueError, match="does not support stereo mode"):
        _StaticDataset(config=config, frames=frames, is_stereo=False)


def test_get_active_camera_roles_returns_left_for_mono(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path), load_stereo=False)
    dataset = _StaticDataset(config=config, frames=frames, is_stereo=False)

    assert dataset.get_active_camera_roles() == ["left"]


def test_get_active_camera_roles_returns_stereo_roles(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "image_path_right": str(tmp_path / "right.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path), load_stereo=True)
    dataset = _StaticDataset(config=config, frames=frames, is_stereo=True)

    assert dataset.get_active_camera_roles() == ["left", "right"]


def test_capability_contract_rejects_duplicate_metadata_destinations(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))
    duplicates = [
        (tmp_path / "a.txt", "dup.txt", True),
        (tmp_path / "b.txt", "dup.txt", False),
    ]

    with pytest.raises(ValueError, match="metadata destination filenames must be unique"):
        _StaticDataset(
            config=config,
            frames=frames,
            metadata_files_with_dest=duplicates,
        )


def test_get_image_paths_rejects_unknown_camera(tmp_path: Path) -> None:
    frames = [
        {
            "image_path": str(tmp_path / "left.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))
    dataset = _StaticDataset(config=config, frames=frames)

    with pytest.raises(ValueError, match="Unsupported camera"):
        dataset.get_image_paths("rear")


def test_depth_lookup_uses_frame_metadata_mapping(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    depth_dir = tmp_path / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    rgb_filename = "1305031102.175304.png"
    depth_filename = "1305031102.160407.png"

    cv2.imwrite(str(rgb_dir / rgb_filename), np.zeros((2, 2, 3), dtype=np.uint8))
    cv2.imwrite(str(depth_dir / depth_filename), np.full((2, 2), 1000, dtype=np.uint16))

    frames = [
        {
            "image_path": str(rgb_dir / rgb_filename),
            "depth_path": str(depth_dir / depth_filename),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))
    dataset = _StaticDataset(config=config, frames=frames, native_depth_dir=depth_dir)

    depth = dataset.load_depth_for_frame(
        rgb_filename=rgb_filename,
        camera="left",
        use_estimated=False,
    )

    assert depth is not None
    assert depth.dtype == np.float32
    assert np.allclose(depth, 1.0)
    assert dataset.native_decode_calls == 1


def test_depth_loader_rejects_multi_channel_depth_image(tmp_path: Path) -> None:
    depth_dir = tmp_path / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    bad_depth_path = depth_dir / "bad_depth.png"
    cv2.imwrite(str(bad_depth_path), np.zeros((2, 2, 3), dtype=np.uint8))

    frames = [
        {
            "image_path": str(tmp_path / "frame.png"),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))
    dataset = _StaticDataset(config=config, frames=frames, native_depth_dir=depth_dir)

    with pytest.raises(RuntimeError, match="single-channel"):
        dataset._load_depth_file(bad_depth_path, is_estimated=False)


def test_estimated_depth_lookup_falls_back_across_candidates(tmp_path: Path) -> None:
    rgb_dir = tmp_path / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    rgb_filename = "frame_000000.png"
    cv2.imwrite(str(rgb_dir / rgb_filename), np.zeros((2, 2, 3), dtype=np.uint8))

    # Create a stale first-priority directory with no matching file.
    (tmp_path / "left_foundation_stereo_depth").mkdir(parents=True, exist_ok=True)

    # Place the actual matching depth in a lower-priority candidate directory.
    fallback_depth_dir = tmp_path / "left_da3_depth"
    fallback_depth_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(fallback_depth_dir / rgb_filename), np.full((2, 2), 512, dtype=np.uint16))

    frames = [
        {
            "image_path": str(rgb_dir / rgb_filename),
            "timestamp": 0.0,
            "sequence_id": "seq",
            "frame_id": 0,
        }
    ]
    config = DatasetConfig(type="mock", path=str(tmp_path))
    dataset = _StaticDataset(config=config, frames=frames)

    depth = dataset.load_depth_for_frame(
        rgb_filename=rgb_filename,
        camera="left",
        use_estimated=True,
    )

    assert depth is not None
    assert depth.dtype == np.float32
    assert np.allclose(depth, 2.0)  # Estimated depth uses value/256.0 encoding
