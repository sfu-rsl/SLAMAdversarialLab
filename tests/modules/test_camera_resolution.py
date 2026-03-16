"""Tests for dataset/canonical camera resolution in module depth setup."""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from slamadverseriallab.config.schema import PerturbationConfig
from slamadverseriallab.datasets.base import CameraIntrinsics
from slamadverseriallab.modules.base import ModuleSetupContext, PerturbationModule


class _DatasetCameraResolutionStub:
    """Dataset stub exposing camera-name APIs used by depth setup."""

    def __init__(
        self,
        left_camera: str,
        right_camera: Optional[str] = None,
        stereo_calibration: Optional[CameraIntrinsics] = None,
        fail_left_camera_lookup: bool = False,
        active_roles: Optional[List[str]] = None,
        is_stereo: Optional[bool] = None,
    ) -> None:
        self._left_camera = left_camera
        self._right_camera = right_camera
        self._stereo_calibration = stereo_calibration or CameraIntrinsics(
            fx=500.0,
            fy=500.0,
            cx=0.0,
            cy=0.0,
            baseline=0.1,
        )
        self._fail_left_camera_lookup = fail_left_camera_lookup
        self._active_roles = list(active_roles) if active_roles is not None else None
        self._is_stereo = (right_camera is not None) if is_stereo is None else is_stereo

    @property
    def is_stereo(self) -> bool:
        return self._is_stereo

    def get_active_camera_roles(self) -> List[str]:
        if self._active_roles is not None:
            return list(self._active_roles)
        roles = ["left"]
        if self._right_camera is not None:
            roles.append("right")
        return roles

    def get_image_directory_name(self, camera: str = "left") -> str:
        if camera == "left" and self._fail_left_camera_lookup:
            raise ValueError("synthetic left camera lookup failure")
        if camera == "right":
            if self._right_camera is None:
                raise ValueError("Mono dataset has no right camera")
            return self._right_camera
        return self._left_camera

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

    def resolve_camera_directories(self, source_root: Path):
        camera_dirs = {}
        for role in self.get_active_camera_roles():
            camera_dirs[role] = self.resolve_camera_directory_name(source_root, role)

        if "right" in camera_dirs and camera_dirs["right"] == camera_dirs["left"]:
            raise RuntimeError(
                f"Resolved left and right cameras to the same directory: {camera_dirs['left']}"
            )

        return camera_dirs

    def get_canonical_camera_name(self, camera: str = "left") -> str:
        return "image_3" if camera == "right" else "image_2"

    def get_camera_intrinsics(self, camera: str = "left") -> CameraIntrinsics:
        return self._stereo_calibration



class _CameraResolutionModule(PerturbationModule):
    """Minimal concrete module for testing camera-resolution helpers."""

    def _setup(self, context: ModuleSetupContext) -> None:
        self._apply_setup_context(context)

    def _apply(self, image: np.ndarray, depth, frame_idx: int, camera: str, **kwargs) -> np.ndarray:
        return image


def _make_module(dataset_stub: _DatasetCameraResolutionStub) -> _CameraResolutionModule:
    module = _CameraResolutionModule(PerturbationConfig(name="camera_resolution", type="none"))
    module._dataset = dataset_stub
    return module


def test_detect_cameras_prefers_dataset_native_when_available(tmp_path: Path) -> None:
    (tmp_path / "rgb").mkdir(parents=True, exist_ok=True)
    (tmp_path / "image_2").mkdir(parents=True, exist_ok=True)

    module = _make_module(_DatasetCameraResolutionStub(left_camera="rgb"))
    cameras = module._detect_cameras(tmp_path)

    assert cameras == ["left"]
    assert module.camera_dirs == {"left": "rgb"}


def test_detect_cameras_fails_when_dataset_native_dir_missing(tmp_path: Path) -> None:
    (tmp_path / "image_2").mkdir(parents=True, exist_ok=True)

    module = _make_module(_DatasetCameraResolutionStub(left_camera="rgb"))

    with pytest.raises(RuntimeError, match="Camera directory 'rgb' for role 'left' not found"):
        module._detect_cameras(tmp_path)


def test_detect_cameras_resolves_stereo_from_dataset_native_dirs(tmp_path: Path) -> None:
    (tmp_path / "mav0" / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mav0" / "cam1" / "data").mkdir(parents=True, exist_ok=True)

    module = _make_module(
        _DatasetCameraResolutionStub(
            left_camera="mav0/cam0/data",
            right_camera="mav0/cam1/data",
        )
    )
    cameras = module._detect_cameras(tmp_path)

    assert cameras == ["left", "right"]
    assert module.camera_dirs == {"left": "mav0/cam0/data", "right": "mav0/cam1/data"}


def test_detect_cameras_uses_active_camera_roles_contract(tmp_path: Path) -> None:
    (tmp_path / "mav0" / "cam0" / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "mav0" / "cam1" / "data").mkdir(parents=True, exist_ok=True)

    module = _make_module(
        _DatasetCameraResolutionStub(
            left_camera="mav0/cam0/data",
            right_camera="mav0/cam1/data",
            active_roles=["left", "right"],
            is_stereo=False,
        )
    )
    cameras = module._detect_cameras(tmp_path)

    assert cameras == ["left", "right"]
    assert module.camera_dirs == {"left": "mav0/cam0/data", "right": "mav0/cam1/data"}


def test_detect_cameras_surfaces_dataset_camera_lookup_errors(tmp_path: Path) -> None:
    (tmp_path / "image_2").mkdir(parents=True, exist_ok=True)

    module = _make_module(
        _DatasetCameraResolutionStub(
            left_camera="image_2",
            fail_left_camera_lookup=True,
        )
    )

    with pytest.raises(ValueError, match="synthetic left camera lookup failure"):
        module._detect_cameras(tmp_path)


def test_foundation_stereo_uses_resolved_camera_dirs(tmp_path: Path) -> None:
    left_dir = tmp_path / "mav0" / "cam0" / "data"
    right_dir = tmp_path / "mav0" / "cam1" / "data"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    module = _make_module(
        _DatasetCameraResolutionStub(
            left_camera="mav0/cam0/data",
            right_camera="mav0/cam1/data",
        )
    )
    module.depth_dirs = {}

    calls = []

    def _fake_run_foundation_stereo(left_images_dir: Path, right_images_dir: Path, output_dir: Path, fx: float, baseline: float):
        calls.append((left_images_dir, right_images_dir, output_dir, fx, baseline))

    module._run_foundation_stereo = _fake_run_foundation_stereo  # type: ignore[method-assign]
    module._depth_cache_complete = lambda image_dir, depth_dir: False  # type: ignore[method-assign]

    module._setup_foundation_stereo_depth(
        source_path=tmp_path,
        dataset=module.dataset,
        cameras=["left", "right"],
    )

    assert len(calls) == 2
    assert calls[0][0] == left_dir
    assert calls[0][1] == right_dir
    assert calls[1][0] == right_dir
    assert calls[1][1] == left_dir
    assert set(module.depth_dirs.keys()) == {"left", "right"}


def test_sensor_depth_loading_fails_fast_when_dataset_returns_none(tmp_path: Path) -> None:
    class _SensorDepthNoneDataset(_DatasetCameraResolutionStub):
        def load_depth_for_frame(self, rgb_filename: str, camera: str = "left", use_estimated: bool = True):
            return None

    module = _make_module(_SensorDepthNoneDataset(left_camera="rgb"))
    module._depth_setup_complete = True
    module._depth_source = "sensor"
    module.depth_dirs = {"left": tmp_path / "depth"}

    with pytest.raises(RuntimeError, match="Sensor depth unavailable for frame '000000.png'"):
        module._load_depth_from_disk(camera="left", rgb_filename="000000.png")
