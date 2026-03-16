"""Tests for module runtime context behavior and source-path handling."""

from pathlib import Path

import numpy as np
import pytest

from slamadverseriallab.config.schema import PerturbationConfig
from slamadverseriallab.datasets.base import CameraIntrinsics
from slamadverseriallab.modules.base import ModuleSetupContext
from slamadverseriallab.modules.transport.video_encoding_base import VideoEncodingModuleBase


class _VideoEncodingStub(VideoEncodingModuleBase):
    """Concrete test stub for video-encoding base behavior."""

    def _setup_encoding_params(self, params: dict) -> None:
        return

    def _get_encoding_flags(self):
        return []

    def _get_log_info(self) -> str:
        return "stub"


def test_video_encoding_base_uses_context_source_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    override_dir = tmp_path / "override"
    source_dir.mkdir(parents=True, exist_ok=True)
    override_dir.mkdir(parents=True, exist_ok=True)

    # Keep test independent from local ffmpeg installation.
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/ffmpeg")

    module = _VideoEncodingStub(PerturbationConfig(name="video_stub", type="none"))
    module.setup(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=source_dir,
            total_frames=2,
            input_path=None,
        )
    )
    assert module._get_source_path() == source_dir
    assert module._setup_complete is True

    module._compressed_frames["left"] = [np.zeros((2, 2, 3), dtype=np.uint8)]

    module.update_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=source_dir,
            total_frames=2,
            input_path=override_dir,
        ),
        reason="test_input_override",
    )

    assert module._get_source_path() == override_dir
    assert module._setup_complete is True
    assert module._compressed_frames == {}


def test_fog_incremental_uses_context_total_frames(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    monkeypatch.setattr(
        FogModule,
        "_setup_depth_estimation",
        lambda self, context=None: None,
    )

    module = FogModule(
        PerturbationConfig(
            name="fog_ctx_frames",
            type="none",
            parameters={
                "incremental": True,
                "start_visibility_m": 120.0,
                "end_visibility_m": 20.0,
                "total_frames": 9999,  # Must be ignored in favor of context.
            },
        )
    )

    module.setup(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=5,
            input_path=None,
        )
    )
    assert module.params.total_frames == 5

    module.update_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=7,
            input_path=None,
        ),
        reason="test_frame_update",
    )
    assert module.params.total_frames == 7


def test_fog_context_update_refreshes_depth_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.fog import FogModule

    calls = []

    def _fake_setup_depth_estimation(self, context=None):
        calls.append(context)
        if len(calls) > 1:
            # Shared refresh helper must clear runtime state before setup.
            assert self._depth_setup_complete is False
            assert self.depth_dirs == {}
        self.depth_dirs = {"left": tmp_path / "image_2_depth"}
        self._depth_setup_complete = True

    monkeypatch.setattr(FogModule, "_setup_depth_estimation", _fake_setup_depth_estimation)

    module = FogModule(
        PerturbationConfig(
            name="fog_refresh",
            type="none",
            parameters={"visibility_m": 30.0},
        )
    )
    module.setup(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=3,
            input_path=None,
        )
    )

    assert len(calls) == 1
    assert module._depth_setup_complete is True
    assert module.depth_dirs == {"left": tmp_path / "image_2_depth"}

    # Seed stale state that must be cleared by refresh helper.
    module.depth_dirs = {"left": tmp_path / "stale_depth"}
    module._depth_setup_complete = True

    module.update_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=4,
            input_path=tmp_path / "override",
        ),
        reason="depth_refresh",
    )
    assert len(calls) == 2
    assert module._depth_setup_complete is True
    assert module.depth_dirs == {"left": tmp_path / "image_2_depth"}


def test_rain_context_update_refreshes_depth_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from slamadverseriallab.modules.scene.rain import RainModule

    calls = []

    def _fake_setup_depth_estimation(self, context=None):
        calls.append(context)
        if len(calls) > 1:
            # Shared refresh helper must clear runtime state before setup.
            assert self._depth_setup_complete is False
            assert self.depth_dirs == {}
        self.depth_dirs = {"left": tmp_path / "image_2_depth"}
        self._depth_setup_complete = True
        self.cameras = ["left"]

    monkeypatch.setattr(RainModule, "_setup_depth_estimation", _fake_setup_depth_estimation)
    monkeypatch.setattr(RainModule, "_ensure_depth_aliases_for_rain", lambda self: None)
    monkeypatch.setattr(RainModule, "_create_dataset_symlink", lambda self: None)

    module = RainModule(
        PerturbationConfig(
            name="rain_refresh",
            type="none",
            parameters={"rain_rendering_path": str(tmp_path / "rain-rendering")},
        )
    )
    module.setup(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=5,
            input_path=None,
        )
    )

    assert len(calls) == 1
    assert module._depth_setup_complete is True
    assert module.depth_dirs == {"left": tmp_path / "image_2_depth"}

    module.depth_dirs = {"left": tmp_path / "stale_depth"}
    module._depth_setup_complete = True

    module.update_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=tmp_path,
            total_frames=6,
            input_path=tmp_path / "override",
        ),
        reason="depth_refresh",
    )
    assert len(calls) == 2
    assert module._depth_setup_complete is True
    assert module.depth_dirs == {"left": tmp_path / "image_2_depth"}


def test_rain_depth_aliases_use_active_depth_directory(tmp_path: Path) -> None:
    from slamadverseriallab.modules.scene.rain import RainModule

    source_path = tmp_path / "sequence"
    source_path.mkdir(parents=True, exist_ok=True)
    active_depth_dir = source_path / "left_da3_depth"
    active_depth_dir.mkdir(parents=True, exist_ok=True)

    module = RainModule(
        PerturbationConfig(
            name="rain_alias_contract",
            type="none",
            parameters={"rain_rendering_path": str(tmp_path / "rain-rendering")},
        )
    )
    module._apply_setup_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=source_path,
            total_frames=1,
            input_path=None,
        )
    )
    module.cameras = ["left"]
    module.depth_dirs = {"left": active_depth_dir}

    module._ensure_depth_aliases_for_rain()

    alias_path = source_path / "left_depth"
    assert alias_path.is_symlink()
    assert alias_path.resolve() == active_depth_dir.resolve()

    # Idempotent call should keep the same valid alias.
    module._ensure_depth_aliases_for_rain()
    assert alias_path.is_symlink()
    assert alias_path.resolve() == active_depth_dir.resolve()


def test_rain_depth_aliases_reject_conflicting_existing_path(tmp_path: Path) -> None:
    from slamadverseriallab.modules.scene.rain import RainModule

    source_path = tmp_path / "sequence"
    source_path.mkdir(parents=True, exist_ok=True)
    active_depth_dir = source_path / "left_da3_depth"
    active_depth_dir.mkdir(parents=True, exist_ok=True)

    module = RainModule(
        PerturbationConfig(
            name="rain_alias_conflict",
            type="none",
            parameters={"rain_rendering_path": str(tmp_path / "rain-rendering")},
        )
    )
    module._apply_setup_context(
        ModuleSetupContext(
            dataset=object(),
            dataset_path=source_path,
            total_frames=1,
            input_path=None,
        )
    )
    module.cameras = ["left"]
    module.depth_dirs = {"left": active_depth_dir}

    conflict_path = source_path / "left_depth"
    conflict_path.mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="already exists"):
        module._ensure_depth_aliases_for_rain()


def test_speed_blur_uses_shared_depth_loader(tmp_path: Path) -> None:
    from slamadverseriallab.modules.optics.speed_blur import SpeedBlurModule

    class _DatasetStub:
        def __init__(self):
            self.path = tmp_path
            self.calls = []
            self.intrinsics_calls = []

        def get_camera_intrinsics(self, camera: str = "left"):
            self.intrinsics_calls.append(camera)
            return CameraIntrinsics(fx=500.0, fy=500.0, cx=2.0, cy=2.0)

        def load_depth_for_frame(self, rgb_filename: str, camera: str, use_estimated: bool):
            self.calls.append((rgb_filename, camera, use_estimated))
            return np.ones((4, 4), dtype=np.float32) * 5.0

    dataset = _DatasetStub()
    module = SpeedBlurModule(PerturbationConfig(name="speed_blur_test", type="none", parameters={}))
    module.setup(
        ModuleSetupContext(
            dataset=dataset,
            dataset_path=tmp_path,
            total_frames=2,
            input_path=None,
        )
    )

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    module._apply(image=image, depth=None, frame_idx=0, camera="left", rgb_filename="000000.png")
    assert dataset.calls == [("000000.png", "left", True)]
    assert dataset.intrinsics_calls == ["left"]


def test_speed_blur_requires_rgb_filename_for_depth_loading(tmp_path: Path) -> None:
    from slamadverseriallab.modules.optics.speed_blur import SpeedBlurModule

    class _DatasetStub:
        path = tmp_path

        def get_camera_intrinsics(self, camera: str = "left"):
            return None

        def load_depth_for_frame(self, rgb_filename: str, camera: str, use_estimated: bool):
            return np.ones((4, 4), dtype=np.float32) * 5.0

    module = SpeedBlurModule(PerturbationConfig(name="speed_blur_test", type="none", parameters={}))
    module.setup(
        ModuleSetupContext(
            dataset=_DatasetStub(),
            dataset_path=tmp_path,
            total_frames=2,
            input_path=None,
        )
    )

    with pytest.raises(ValueError, match="rgb_filename is required for speed_blur module to load depth maps"):
        module._apply(
            image=np.zeros((4, 4, 3), dtype=np.uint8),
            depth=None,
            frame_idx=0,
            camera="left",
        )


def test_speed_blur_caches_intrinsics_per_camera(tmp_path: Path) -> None:
    from slamadverseriallab.modules.optics.speed_blur import SpeedBlurModule

    class _DatasetStub:
        def __init__(self):
            self.path = tmp_path
            self.depth_calls = []
            self.intrinsics_calls = []

        def get_camera_intrinsics(self, camera: str = "left"):
            self.intrinsics_calls.append(camera)
            if camera == "left":
                return CameraIntrinsics(fx=500.0, fy=500.0, cx=2.0, cy=2.0)
            if camera == "right":
                return CameraIntrinsics(fx=700.0, fy=700.0, cx=1.5, cy=1.5)
            return None

        def load_depth_for_frame(self, rgb_filename: str, camera: str, use_estimated: bool):
            self.depth_calls.append((rgb_filename, camera, use_estimated))
            return np.ones((4, 4), dtype=np.float32) * 5.0

    dataset = _DatasetStub()
    module = SpeedBlurModule(PerturbationConfig(name="speed_blur_test", type="none", parameters={}))
    module.setup(
        ModuleSetupContext(
            dataset=dataset,
            dataset_path=tmp_path,
            total_frames=4,
            input_path=None,
        )
    )

    image = np.zeros((4, 4, 3), dtype=np.uint8)
    module._apply(image=image, depth=None, frame_idx=0, camera="left", rgb_filename="000000.png")
    module._apply(image=image, depth=None, frame_idx=1, camera="left", rgb_filename="000001.png")
    module._apply(image=image, depth=None, frame_idx=2, camera="right", rgb_filename="000002.png")
    module._apply(image=image, depth=None, frame_idx=3, camera="right", rgb_filename="000003.png")

    assert dataset.intrinsics_calls == ["left", "right"]
    assert module._intrinsics_by_camera["left"].fx == pytest.approx(500.0)
    assert module._intrinsics_by_camera["right"].fx == pytest.approx(700.0)
    assert dataset.depth_calls == [
        ("000000.png", "left", True),
        ("000001.png", "left", True),
        ("000002.png", "right", True),
        ("000003.png", "right", True),
    ]
