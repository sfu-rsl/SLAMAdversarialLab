"""Tests for strict pipeline failure behavior on frame/module exceptions."""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import DatasetConfig, ExperimentConfig, OutputConfig
from slamadverseriallab.core.pipeline import Pipeline


class _DummyDataset:
    """Minimal iterable dataset for pipeline unit tests."""

    def __init__(self, num_frames: int = 3):
        self._frames: List[Dict[str, Any]] = []
        self.is_stereo = False
        for i in range(num_frames):
            self._frames.append(
                {
                    "image": np.zeros((4, 4, 3), dtype=np.uint8),
                    "depth": None,
                    "timestamp": float(i),
                    "sequence_id": "seq",
                    "frame_id": i,
                    "rgb_filename": f"{i:06d}.png",
                }
            )

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self):
        return iter(self._frames)

    def get_image_directory_name(self, camera: str = "left") -> str:
        return "image_2" if camera == "left" else "image_3"

    def get_active_camera_roles(self) -> List[str]:
        return ["left"]

    def filter_metadata_for_kept_frames(self, output_dir: Path, kept_frame_indices: List[int], total_frames: int) -> None:
        # Not needed for these tests.
        return


class _FailingModule:
    """Test module that raises on a selected frame."""

    def __init__(self, name: str = "failing_module", fail_on_frame: int = 0):
        self.name = name
        self.enabled = True
        self.fail_on_frame = fail_on_frame
        self.calls: List[int] = []

    def apply(self, image: np.ndarray, depth: Any, frame_idx: int, camera: str, **kwargs) -> np.ndarray:
        self.calls.append(frame_idx)
        if frame_idx == self.fail_on_frame:
            raise ValueError("synthetic failure")
        return image

    def cleanup(self) -> None:
        return

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "test"}


class _DropFrameModule:
    """Test module that intentionally drops one frame by returning None."""

    def __init__(self, name: str = "drop_module", drop_frame: int = 1):
        self.name = name
        self.enabled = True
        self.drop_frame = drop_frame
        self.calls: List[int] = []

    def apply(self, image: np.ndarray, depth: Any, frame_idx: int, camera: str, **kwargs) -> Any:
        self.calls.append(frame_idx)
        if frame_idx == self.drop_frame:
            return None
        return image

    def cleanup(self) -> None:
        return

    def get_info(self) -> Dict[str, Any]:
        return {"name": self.name, "type": "test"}


class _DatasetWithoutDropMetadataSupport(_DummyDataset):
    """Dataset stub that explicitly does not support metadata filtering after drops."""

    def filter_metadata_for_kept_frames(
        self,
        output_dir: Path,
        kept_frame_indices: List[int],
        total_frames: int,
    ) -> None:
        raise NotImplementedError("drop metadata filtering not supported")


def _make_config(tmp_path: Path) -> Config:
    return Config(
        experiment=ExperimentConfig(name="pipeline_failure_mode_test"),
        dataset=DatasetConfig(type="mock", path="mock"),
        perturbations=[],
        output=OutputConfig(
            base_dir=str(tmp_path),
            save_images=False,
            create_timestamp_dir=False,
        ),
    )


def _make_pipeline(tmp_path: Path, dataset: _DummyDataset, modules: List[Any]) -> Pipeline:
    pipeline = Pipeline(_make_config(tmp_path))
    pipeline.dataset = dataset
    pipeline.modules = modules
    pipeline.output_dir = tmp_path / "out"
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.output_manager = None
    return pipeline


def test_pipeline_fails_fast_on_first_frame_exception(tmp_path: Path) -> None:
    dataset = _DummyDataset(num_frames=3)
    failing_module = _FailingModule(fail_on_frame=0)
    pipeline = _make_pipeline(tmp_path, dataset, [failing_module])

    with pytest.raises(RuntimeError, match=r"Error processing frame 0 with module failing_module"):
        pipeline.run()

    # Fail-fast: module should not be called for later frames.
    assert failing_module.calls == [0]


def test_pipeline_allows_intentional_dropped_frames(tmp_path: Path) -> None:
    dataset = _DummyDataset(num_frames=3)
    drop_module = _DropFrameModule(drop_frame=1)
    pipeline = _make_pipeline(tmp_path, dataset, [drop_module])

    results = pipeline.run()

    assert results["status"] == "success"
    assert results["frames_processed"] == 3
    assert results["module_stats"]["drop_module"]["dropped_frames"] == [1]
    assert drop_module.calls == [0, 1, 2]


def test_pipeline_fails_when_dataset_does_not_support_drop_metadata_filtering(tmp_path: Path) -> None:
    dataset = _DatasetWithoutDropMetadataSupport(num_frames=3)
    drop_module = _DropFrameModule(drop_frame=1)
    pipeline = _make_pipeline(tmp_path, dataset, [drop_module])

    with pytest.raises(NotImplementedError, match="drop metadata filtering not supported"):
        pipeline.run()


def test_load_modules_requires_loaded_dataset_context(tmp_path: Path) -> None:
    pipeline = Pipeline(_make_config(tmp_path))
    pipeline.dataset = None

    with pytest.raises(RuntimeError, match="Dataset must be loaded before module setup"):
        pipeline.load_modules()
