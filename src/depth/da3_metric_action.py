"""Depth Anything 3 metric-depth action.

This helper runs DA3Metric inference for a camera image directory and writes
depth caches aligned to RGB filenames.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..datasets.base import CameraIntrinsics


class DA3MetricAction:
    """Run DA3Metric depth inference and write PNG/NPZ depth caches."""

    def __init__(
        self,
        *,
        model_id: str = "depth-anything/DA3METRIC-LARGE",
        device: str = "auto",
        process_res: int = 504,
        fallback_hfov_deg: float = 70.0,
        save_npz: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.process_res = int(process_res)
        self.fallback_hfov_deg = float(fallback_hfov_deg)
        self.save_npz = bool(save_npz)
        self._model = None

    @staticmethod
    def _collect_image_files(image_dir: Path) -> List[Path]:
        image_files = sorted(image_dir.glob("*.png"))
        if image_files:
            return image_files
        return sorted(image_dir.glob("*.jpg"))

    def _load_model(self):
        """Load DA3 model lazily to keep import-time dependencies optional."""
        if self._model is not None:
            return self._model

        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as e:
            raise RuntimeError(
                "Depth Anything 3 runtime is unavailable. Install with:\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n"
                "and ensure torch is available."
            ) from e

        model = DepthAnything3.from_pretrained(self.model_id)

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device

        model = model.to(device).eval()
        logger.info("Loaded DA3 model '%s' on device=%s", self.model_id, device)
        self._model = model
        return model

    def _resolve_focal_px(
        self,
        intrinsics: Optional[CameraIntrinsics],
        width: int,
        height: int,
    ) -> Tuple[float, str]:
        """Resolve focal length in pixels for DA3Metric scaling."""
        if intrinsics is not None:
            fx = float(intrinsics.fx)
            fy = float(intrinsics.fy)

            # If calibration resolution is known, rescale intrinsics to frame resolution.
            if intrinsics.width is not None and float(intrinsics.width) > 0.0:
                fx *= float(width) / float(intrinsics.width)
            if intrinsics.height is not None and float(intrinsics.height) > 0.0:
                fy *= float(height) / float(intrinsics.height)

            values = np.asarray([fx, fy], dtype=np.float32)
            if np.all(np.isfinite(values)) and fx > 0.0 and fy > 0.0:
                return float((fx + fy) * 0.5), "dataset_intrinsics"

        # Fallback estimate from horizontal FoV.
        focal = width / (2.0 * np.tan(np.radians(self.fallback_hfov_deg * 0.5)))
        if not np.isfinite(focal) or focal <= 0:
            raise RuntimeError(
                f"Invalid fallback focal estimate for width={width}, height={height}, "
                f"hfov_deg={self.fallback_hfov_deg}"
            )
        return float(focal), "estimated_hfov"

    @staticmethod
    def _extract_depth_prediction(prediction: object, image_path: Path) -> np.ndarray:
        """Extract single-image depth map from a DA3 prediction object."""
        if not hasattr(prediction, "depth"):
            raise RuntimeError(f"DA3 prediction does not include depth for image: {image_path}")

        depth_pred = np.asarray(prediction.depth)
        if depth_pred.ndim == 2:
            return depth_pred.astype(np.float32)
        if depth_pred.ndim == 3 and depth_pred.shape[0] == 1:
            return depth_pred[0].astype(np.float32)

        raise RuntimeError(
            f"Unexpected DA3 depth shape for {image_path}: {depth_pred.shape}"
        )

    @staticmethod
    def _scale_focal_for_prediction(
        focal_px: float,
        *,
        image_width: int,
        image_height: int,
        depth_width: int,
        depth_height: int,
    ) -> float:
        """Scale focal to the model output resolution used for this depth prediction."""
        if image_width <= 0 or image_height <= 0:
            raise RuntimeError(
                f"Invalid image size for focal scaling: {image_width}x{image_height}"
            )
        if depth_width <= 0 or depth_height <= 0:
            raise RuntimeError(
                f"Invalid depth size for focal scaling: {depth_width}x{depth_height}"
            )

        scale_x = float(depth_width) / float(image_width)
        scale_y = float(depth_height) / float(image_height)
        scaled_focal = float(focal_px) * ((scale_x + scale_y) * 0.5)
        if not np.isfinite(scaled_focal) or scaled_focal <= 0.0:
            raise RuntimeError(
                "Invalid scaled focal from frame/depth dimensions: "
                f"focal={focal_px}, image={image_width}x{image_height}, "
                f"depth={depth_width}x{depth_height}"
            )
        return scaled_focal

    def run_sequence(
        self,
        *,
        image_dir: Path,
        output_dir: Path,
        camera_role: str,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> None:
        """Run DA3Metric on all images in ``image_dir`` and write depth outputs."""
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = self._collect_image_files(image_dir)
        if not image_files:
            raise RuntimeError(f"No PNG/JPG images found for DA3 inference in: {image_dir}")

        model = self._load_model()

        written = 0
        for image_path in image_files:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Failed to read image for DA3 inference: {image_path}")
            image_h, image_w = image.shape[:2]

            focal_px, focal_source = self._resolve_focal_px(intrinsics, image_w, image_h)

            prediction = model.inference(
                [str(image_path)],
                process_res=self.process_res,
                export_dir=None,
                export_format="mini_npz",
            )
            depth_pred = self._extract_depth_prediction(prediction, image_path)
            depth_h, depth_w = depth_pred.shape[:2]

            focal_for_prediction = self._scale_focal_for_prediction(
                focal_px,
                image_width=image_w,
                image_height=image_h,
                depth_width=depth_w,
                depth_height=depth_h,
            )

            # Official DA3Metric conversion:
            # metric_depth = focal * net_output / 300
            depth_metric = (depth_pred.astype(np.float32) * focal_for_prediction) / 300.0
            if depth_metric.shape != (image_h, image_w):
                depth_metric = cv2.resize(
                    depth_metric,
                    (image_w, image_h),
                    interpolation=cv2.INTER_LINEAR,
                )
            depth_metric = np.where(np.isfinite(depth_metric), np.maximum(depth_metric, 0.0), 0.0)

            if written == 0:
                logger.info(
                    "DA3 focal for camera=%s: %.3fpx (source=%s) -> scaled=%.3fpx at %dx%d model depth",
                    camera_role,
                    focal_px,
                    focal_source,
                    focal_for_prediction,
                    depth_w,
                    depth_h,
                )

            depth_png = np.clip(depth_metric * 256.0, 0.0, 65535.0).astype(np.uint16)
            png_path = output_dir / image_path.name
            if not cv2.imwrite(str(png_path), depth_png):
                raise RuntimeError(f"Failed to write DA3 depth PNG: {png_path}")

            if self.save_npz:
                npz_path = output_dir / f"{image_path.stem}.npz"
                np.savez_compressed(npz_path, depth=depth_metric.astype(np.float32))

            written += 1

        if written != len(image_files):
            raise RuntimeError(
                f"DA3 depth output incomplete for camera={camera_role}: "
                f"wrote {written}/{len(image_files)} files"
            )

        logger.info(
            "DA3 depth generation complete for camera=%s (%d frames) -> %s",
            camera_role,
            written,
            output_dir,
        )
