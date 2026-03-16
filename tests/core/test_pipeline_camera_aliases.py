"""Tests for standardized output camera aliases."""

import os
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

from slamadverseriallab.config.parser import Config
from slamadverseriallab.config.schema import DatasetConfig, ExperimentConfig, OutputConfig
from slamadverseriallab.core.pipeline import Pipeline


class _DatasetCameraStub:
    """Dataset stub exposing configurable camera directory names."""

    def __init__(
        self,
        left_camera: str,
        right_camera: Optional[str] = None,
        active_roles: Optional[List[str]] = None,
        is_stereo: Optional[bool] = None,
    ):
        self._left_camera = left_camera
        self._right_camera = right_camera
        self._active_roles = list(active_roles) if active_roles is not None else None
        self.is_stereo = (right_camera is not None) if is_stereo is None else is_stereo

    def has_depth(self) -> bool:
        return False

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        return []

    def get_active_camera_roles(self) -> List[str]:
        if self._active_roles is not None:
            return list(self._active_roles)
        roles = ["left"]
        if self._right_camera is not None:
            roles.append("right")
        return roles

    def get_image_directory_name(self, camera: str = "left") -> str:
        if camera == "right":
            if self._right_camera is None:
                raise ValueError("Dataset is mono; right camera unavailable")
            return self._right_camera
        return self._left_camera

    def get_canonical_camera_name(self, camera: str = "left") -> str:
        return "image_3" if camera == "right" else "image_2"


class _ModuleStub:
    """Minimal enabled module for output setup tests."""

    def __init__(self, name: str):
        self.name = name
        self.enabled = True


def _make_pipeline(tmp_path: Path) -> Pipeline:
    config = Config(
        experiment=ExperimentConfig(name="camera_alias_test"),
        dataset=DatasetConfig(type="kitti", path=str(tmp_path), max_frames=2),
        perturbations=[],
        output=OutputConfig(
            base_dir=str(tmp_path / "results"),
            save_images=True,
            create_timestamp_dir=False,
        ),
    )
    pipeline = Pipeline(config)
    pipeline.output_dir = tmp_path / "output"
    pipeline.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline.modules = [_ModuleStub("camera_alias_module")]
    return pipeline


def _assert_symlink_target(alias_path: Path, expected_target: Path) -> None:
    assert alias_path.is_symlink()
    resolved_target = (alias_path.parent / os.readlink(alias_path)).resolve()
    assert resolved_target == expected_target.resolve()


def test_setup_output_manager_creates_mono_camera_alias(tmp_path: Path) -> None:
    dataset_stub = _DatasetCameraStub(left_camera="rgb")
    pipeline = _make_pipeline(tmp_path)
    pipeline.dataset = dataset_stub
    pipeline._setup_output_manager()

    module_dir = pipeline.output_dir / "images" / "camera_alias_module"
    alias_path = module_dir / "rgb"
    canonical_target = module_dir / "image_2"

    _assert_symlink_target(alias_path, canonical_target)


def test_setup_output_manager_creates_nested_stereo_camera_aliases(tmp_path: Path) -> None:
    dataset_stub = _DatasetCameraStub(
        left_camera="mav0/cam0/data",
        right_camera="mav0/cam1/data",
    )
    pipeline = _make_pipeline(tmp_path)
    pipeline.dataset = dataset_stub
    pipeline._setup_output_manager()

    module_dir = pipeline.output_dir / "images" / "camera_alias_module"
    left_alias = module_dir / "mav0" / "cam0" / "data"
    right_alias = module_dir / "mav0" / "cam1" / "data"

    _assert_symlink_target(left_alias, module_dir / "image_2")
    _assert_symlink_target(right_alias, module_dir / "image_3")


def test_setup_output_manager_skips_alias_when_dataset_names_are_canonical(
    tmp_path: Path,
) -> None:
    dataset_stub = _DatasetCameraStub(left_camera="image_2", right_camera="image_3")
    pipeline = _make_pipeline(tmp_path)
    pipeline.dataset = dataset_stub
    pipeline._setup_output_manager()

    module_dir = pipeline.output_dir / "images" / "camera_alias_module"
    assert (module_dir / "image_2").is_dir()
    assert (module_dir / "image_3").is_dir()
    assert not (module_dir / "rgb").exists()
    assert not (module_dir / "mav0").exists()


def test_setup_output_manager_uses_active_camera_roles_contract(tmp_path: Path) -> None:
    dataset_stub = _DatasetCameraStub(
        left_camera="mav0/cam0/data",
        right_camera="mav0/cam1/data",
        active_roles=["left", "right"],
        is_stereo=False,
    )
    pipeline = _make_pipeline(tmp_path)
    pipeline.dataset = dataset_stub
    pipeline._setup_output_manager()

    module_dir = pipeline.output_dir / "images" / "camera_alias_module"
    right_alias = module_dir / "mav0" / "cam1" / "data"

    _assert_symlink_target(right_alias, module_dir / "image_3")


def test_setup_output_manager_requires_loaded_dataset(tmp_path: Path) -> None:
    pipeline = _make_pipeline(tmp_path)
    with pytest.raises(RuntimeError, match="Dataset must be loaded before output manager setup"):
        pipeline._setup_output_manager()
