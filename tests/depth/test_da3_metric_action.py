"""Tests for DA3MetricAction depth generation and focal scaling."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from slamadverseriallab.datasets.base import CameraIntrinsics
from slamadverseriallab.depth.da3_metric_action import DA3MetricAction


class _FakeDA3Model:
    """Simple fake DA3 model that returns pre-seeded depth predictions."""

    def __init__(self, depth_predictions: list[np.ndarray]) -> None:
        self._depth_predictions = list(depth_predictions)
        self.calls = []

    def inference(self, image_paths, process_res, export_dir, export_format):
        self.calls.append(
            {
                "image_paths": tuple(image_paths),
                "process_res": process_res,
                "export_dir": export_dir,
                "export_format": export_format,
            }
        )
        if not self._depth_predictions:
            raise AssertionError("Fake model inference called more times than expected")
        return SimpleNamespace(depth=self._depth_predictions.pop(0))


def _write_image(path: Path, height: int, width: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_da3_metric_action_writes_png_and_npz_with_intrinsics_scaling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "image_2"
    output_dir = tmp_path / "left_da3_depth"
    image_path = image_dir / "000000.png"
    _write_image(image_path, height=4, width=6)

    # Simulate model output at a lower internal resolution (2x3).
    fake_model = _FakeDA3Model(
        depth_predictions=[np.full((1, 2, 3), 10.0, dtype=np.float32)]
    )
    action = DA3MetricAction(save_npz=True)
    monkeypatch.setattr(action, "_load_model", lambda: fake_model)

    intrinsics = CameraIntrinsics(
        fx=600.0,
        fy=600.0,
        cx=3.0,
        cy=2.0,
        width=6.0,
        height=4.0,
    )

    action.run_sequence(
        image_dir=image_dir,
        output_dir=output_dir,
        camera_role="left",
        intrinsics=intrinsics,
    )

    depth_png = cv2.imread(str(output_dir / "000000.png"), cv2.IMREAD_ANYDEPTH)
    assert depth_png is not None
    assert depth_png.shape == (4, 6)
    # Focal scales with 0.5 resize -> effective focal=300, so metric depth=10m.
    assert np.all(depth_png == 2560)

    npz = np.load(output_dir / "000000.npz")
    depth = npz["depth"]
    assert depth.dtype == np.float32
    assert depth.shape == (4, 6)
    assert np.allclose(depth, 10.0, atol=1e-4)

    assert len(fake_model.calls) == 1
    assert fake_model.calls[0]["image_paths"] == (str(image_path),)
    assert fake_model.calls[0]["export_format"] == "mini_npz"


def test_da3_metric_action_uses_fallback_focal_when_intrinsics_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "rgb"
    output_dir = tmp_path / "left_da3_depth"
    image_path = image_dir / "frame_000000.png"
    _write_image(image_path, height=4, width=8)

    fake_model = _FakeDA3Model(
        depth_predictions=[np.full((1, 4, 8), 75.0, dtype=np.float32)]
    )
    action = DA3MetricAction(fallback_hfov_deg=90.0, save_npz=False)
    monkeypatch.setattr(action, "_load_model", lambda: fake_model)

    action.run_sequence(
        image_dir=image_dir,
        output_dir=output_dir,
        camera_role="left",
        intrinsics=None,
    )

    # hfov=90 and width=8 -> focal=4px. depth = 75 * 4 / 300 = 1m.
    depth_png = cv2.imread(str(output_dir / "frame_000000.png"), cv2.IMREAD_ANYDEPTH)
    assert depth_png is not None
    assert np.all(depth_png == 256)
    assert not (output_dir / "frame_000000.npz").exists()


def test_da3_metric_action_rejects_empty_image_directory(tmp_path: Path) -> None:
    image_dir = tmp_path / "empty"
    output_dir = tmp_path / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)

    action = DA3MetricAction()
    with pytest.raises(RuntimeError, match="No PNG/JPG images found"):
        action.run_sequence(
            image_dir=image_dir,
            output_dir=output_dir,
            camera_role="left",
            intrinsics=None,
        )
