"""Tests for strict composite batch preparation context and camera handling."""

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from slamadverseriallab.config.schema import PerturbationConfig
from slamadverseriallab.modules.base import (
    CompositeModule,
    ModuleSetupContext,
    PerturbationModule,
)


def _write_dummy_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    cv2.imwrite(str(path), image)


class _DatasetStub:
    """Minimal iterable dataset stub for composite batch tests."""

    def __init__(
        self,
        frames: List[Dict[str, Any]],
        left_camera: str = "image_2",
        right_camera: Optional[str] = None,
        fail_left_camera_lookup: bool = False,
    ) -> None:
        self._frames = frames
        self._left_camera = left_camera
        self._right_camera = right_camera
        self._fail_left_camera_lookup = fail_left_camera_lookup

    @property
    def is_stereo(self) -> bool:
        return self._right_camera is not None

    @property
    def supports_stereo(self) -> bool:
        return self._right_camera is not None

    def get_active_camera_roles(self) -> List[str]:
        roles = ["left"]
        if self._right_camera is not None:
            roles.append("right")
        return roles

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._frames)

    def get_image_directory_name(self, camera: str = "left") -> str:
        if camera == "left":
            if self._fail_left_camera_lookup:
                raise ValueError("synthetic left camera lookup failure")
            return self._left_camera
        if camera == "right":
            if self._right_camera is None:
                raise ValueError("right camera unavailable")
            return self._right_camera
        raise ValueError(f"Unsupported camera role: {camera}")

    def resolve_camera_directory_name(self, source_root: Path, camera: str = "left") -> str:
        directory_name = self.get_image_directory_name(camera)
        if not directory_name:
            raise ValueError(f"empty camera directory for role '{camera}'")

        resolved = source_root / directory_name
        if not resolved.exists() or not resolved.is_dir():
            raise RuntimeError(
                f"Camera directory '{directory_name}' for role '{camera}' not found under {source_root}."
            )

        return directory_name

    def resolve_camera_directories(self, source_root: Path) -> Dict[str, str]:
        camera_dirs: Dict[str, str] = {}
        for role in self.get_active_camera_roles():
            camera_dirs[role] = self.resolve_camera_directory_name(source_root, role)

        if "right" in camera_dirs and camera_dirs["right"] == camera_dirs["left"]:
            raise RuntimeError(
                f"Resolved left and right cameras to the same directory: {camera_dirs['left']}"
            )

        return camera_dirs


class _NoopModule(PerturbationModule):
    """Minimal concrete module used for composite tests."""

    requires_full_sequence = False

    def _setup(self, context: ModuleSetupContext) -> None:
        self._apply_setup_context(context)

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs,
    ) -> np.ndarray:
        return image


def _make_frames(stereo: bool = False) -> List[Dict[str, Any]]:
    frame: Dict[str, Any] = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "depth": None,
        "timestamp": 0.0,
        "sequence_id": "seq",
        "frame_id": 0,
        "rgb_filename": "000000.png",
    }
    if stereo:
        frame["image_right"] = np.zeros((4, 4, 3), dtype=np.uint8)
        frame["rgb_filename_right"] = "000000.png"
    return [frame]


def _make_composite(
    dataset: Optional[_DatasetStub],
    dataset_path: Path,
) -> Tuple[CompositeModule, _NoopModule]:
    preceding = _NoopModule(PerturbationConfig(name="pre", type="none"))
    batch = _NoopModule(PerturbationConfig(name="batch", type="none"))
    batch.requires_full_sequence = True

    composite = CompositeModule(
        PerturbationConfig(name="composite_batch", type="composite"),
        modules=[preceding, batch],
    )

    context = ModuleSetupContext(
        dataset=dataset,
        dataset_path=dataset_path,
        total_frames=len(dataset) if dataset is not None else 1,
        input_path=None,
    )
    composite.setup(context)
    return composite, batch


def test_prepare_batch_module_requires_loaded_dataset_context(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    composite, batch = _make_composite(dataset=None, dataset_path=tmp_path)

    try:
        with pytest.raises(RuntimeError, match="without a loaded dataset object in setup context"):
            composite._prepare_batch_module(1, batch, depth=None, camera="left")
    finally:
        composite.cleanup()


def test_prepare_batch_module_fails_when_camera_directory_missing(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    dataset = _DatasetStub(
        frames=_make_frames(stereo=True),
        left_camera="image_2",
        right_camera="image_3",
    )
    composite, batch = _make_composite(dataset=dataset, dataset_path=tmp_path)

    try:
        with pytest.raises(RuntimeError, match="Camera directory 'image_3'.*not found"):
            composite._prepare_batch_module(1, batch, depth=None, camera="left")
    finally:
        composite.cleanup()


def test_prepare_batch_module_surfaces_left_camera_contract_errors(tmp_path: Path) -> None:
    _write_dummy_png(tmp_path / "image_2" / "000000.png")
    dataset = _DatasetStub(
        frames=_make_frames(stereo=False),
        left_camera="image_2",
        fail_left_camera_lookup=True,
    )
    composite, batch = _make_composite(dataset=dataset, dataset_path=tmp_path)

    try:
        with pytest.raises(ValueError, match="synthetic left camera lookup failure"):
            composite._prepare_batch_module(1, batch, depth=None, camera="left")
    finally:
        composite.cleanup()
