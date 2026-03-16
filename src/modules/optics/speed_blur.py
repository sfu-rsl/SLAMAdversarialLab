"""Speed-based motion blur module using dataset depth.

Simulates forward-motion blur using depth-aware 3D reprojection.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from ..base import PerturbationModule
from ...robustness.param_spec import BoundaryParamSpec
from ...utils import get_logger

if TYPE_CHECKING:
    from ...datasets.base import CameraIntrinsics as DatasetCameraIntrinsics

logger = get_logger(__name__)


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)

    @classmethod
    def from_projection_matrix(cls, P: np.ndarray) -> "CameraIntrinsics":
        """Extract intrinsics from 3x4 projection matrix."""
        return cls(fx=P[0, 0], fy=P[1, 1], cx=P[0, 2], cy=P[1, 2])

    @classmethod
    def from_image_size(cls, width: int, height: int, hfov_deg: float = 70.0) -> "CameraIntrinsics":
        """Estimate intrinsics from image size assuming horizontal FOV."""
        fx = width / (2 * np.tan(np.radians(hfov_deg / 2)))
        fy = fx  # Assume square pixels
        cx = width / 2
        cy = height / 2
        return cls(fx=fx, fy=fy, cx=cx, cy=cy)


@dataclass
class SpeedBlurParameters:
    """Parameters for speed-based motion blur."""

    speed: float = 50.0
    """Vehicle speed in km/h"""

    exposure_time: Optional[float] = None
    """Exposure time in seconds (if None, computed from fps)"""

    fps: float = 10.0
    """Frame rate - used to compute exposure time if not specified"""

    min_samples: int = 3
    """Minimum number of samples along blur vector"""

    max_samples: int = 31
    """Maximum number of samples along blur vector"""

    samples_per_pixel: float = 0.5
    """Samples per pixel of blur length (adaptive sampling)"""

    depth_scale: float = 1.0
    """Scale factor for depth values (use if depth is not in meters)"""

    blur_scale: float = 1.0
    """Optional scale factor to amplify/reduce blur for artistic effect"""

    max_blur_px: float = 150.0
    """Maximum blur length in pixels (clamp extreme values)"""


class SpeedBlurModule(PerturbationModule):
    """Depth-aware forward motion blur using 3D reprojection."""

    module_name = "speed_blur"
    module_description = "Physically-based forward motion blur using 3D reprojection"

    PARAMETERS_CLASS = SpeedBlurParameters
    SEARCHABLE_PARAMS = {
        "speed": BoundaryParamSpec(domain="continuous"),
    }

    def _setup(self, context) -> None:
        """Setup speed blur module."""
        params = self.parameters or {}

        self.speed_kmh = params.get('speed', 50.0)
        self.speed_ms = self.speed_kmh / 3.6  # Convert to m/s

        self.exposure_time = params.get('exposure_time', None)
        self.fps = params.get('fps', 10.0)

        # If exposure_time not specified, derive from fps
        if self.exposure_time is None:
            self.exposure_time = 1.0 / self.fps

        # Adaptive sampling parameters
        self.min_samples = params.get('min_samples', 3)
        self.max_samples = params.get('max_samples', 31)
        self.samples_per_pixel = params.get('samples_per_pixel', 0.5)

        self.depth_scale = params.get('depth_scale', 1.0)
        self.blur_scale = params.get('blur_scale', 1.0)
        self.max_blur_px = params.get('max_blur_px', 150.0)

        # Camera intrinsics cache by logical camera role.
        self._intrinsics_by_camera: Dict[str, CameraIntrinsics] = {}

        # Precompute camera displacement during exposure
        self.delta_z = self.speed_ms * self.exposure_time

        logger.info(f"SpeedBlurModule '{self.name}' initialized")
        logger.info(f"  Speed: {self.speed_kmh} km/h ({self.speed_ms:.2f} m/s)")
        logger.info(f"  Exposure time: {self.exposure_time * 1000:.1f} ms")
        logger.info(f"  Camera displacement: {self.delta_z * 100:.2f} cm")
        logger.info(f"  Adaptive samples: {self.min_samples}-{self.max_samples}")
        if self.blur_scale != 1.0:
            logger.info(f"  Blur scale: {self.blur_scale}x (non-physical)")

    def _resolve_intrinsics_from_dataset(self, camera: str) -> Optional[CameraIntrinsics]:
        """Resolve camera intrinsics from dataset adapter contract."""
        if self.dataset is None:
            return None

        # Import lazily to avoid registry-time import cycles during module discovery.
        from ...datasets.base import CameraIntrinsics as DatasetCameraIntrinsics

        try:
            raw_intrinsics = self.dataset.get_camera_intrinsics(camera)
        except Exception as e:
            logger.warning(
                f"Failed to get dataset intrinsics for camera={camera}: {e}"
            )
            return None

        if raw_intrinsics is None:
            return None

        if not isinstance(raw_intrinsics, DatasetCameraIntrinsics):
            logger.warning(
                f"Dataset intrinsics for camera={camera} must be CameraIntrinsics; "
                f"got {type(raw_intrinsics).__name__}. Falling back to image-size estimation."
            )
            return None

        try:
            fx = float(raw_intrinsics.fx)
            fy = float(raw_intrinsics.fy)
            cx = float(raw_intrinsics.cx)
            cy = float(raw_intrinsics.cy)
        except (TypeError, ValueError) as e:
            logger.warning(
                f"Dataset intrinsics for camera={camera} are invalid ({e}); "
                "falling back to image-size estimation."
            )
            return None

        if fx <= 0 or fy <= 0 or not np.isfinite([fx, fy, cx, cy]).all():
            logger.warning(
                f"Dataset intrinsics for camera={camera} are non-physical "
                f"(fx={fx}, fy={fy}, cx={cx}, cy={cy}); "
                "falling back to image-size estimation."
            )
            return None

        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def _get_intrinsics_for_camera(self, camera: str, width: int, height: int) -> CameraIntrinsics:
        """Get and cache camera intrinsics per logical camera role."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )

        cached = self._intrinsics_by_camera.get(camera)
        if cached is not None:
            return cached

        intrinsics = self._resolve_intrinsics_from_dataset(camera)
        if intrinsics is not None:
            self._intrinsics_by_camera[camera] = intrinsics
            logger.info(
                f"Loaded intrinsics from dataset for camera={camera}: "
                f"fx={intrinsics.fx:.1f}, fy={intrinsics.fy:.1f}, "
                f"cx={intrinsics.cx:.1f}, cy={intrinsics.cy:.1f}"
            )
            return intrinsics

        estimated = CameraIntrinsics.from_image_size(width, height)
        self._intrinsics_by_camera[camera] = estimated
        logger.warning(
            f"Dataset intrinsics unavailable for camera={camera}; "
            f"estimating from image size: fx={estimated.fx:.1f}, fy={estimated.fy:.1f}"
        )
        return estimated

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply speed-based motion blur using 3D reprojection.

        Args:
            image: Input image (H, W, 3) RGB uint8
            depth: Depth from dataset (may be None, will load from disk)
            frame_idx: Frame index
            camera: Camera identifier
            **kwargs: Additional context (rgb_filename required for depth lookup)

        Returns:
            Motion-blurred image

        Raises:
            RuntimeError: If dataset not set
            ValueError: If rgb_filename is not provided or depth not available
        """
        h, w = image.shape[:2]
        intrinsics = self._get_intrinsics_for_camera(camera, width=w, height=h)

        depth_loaded, rgb_filename = self._load_depth_for_apply(
            camera,
            kwargs,
            provided_depth=depth,
            prefer_dataset=True,
            module_label="speed_blur",
            rgb_filename_reason="load depth maps",
            missing_dataset_error=(
                f"SpeedBlurModule '{self.name}' requires dataset to be set. "
                "Ensure setup(context) includes the dataset instance."
            ),
            missing_depth_error=lambda filename, cam: (
                f"No depth available for frame {filename} (camera={cam}). "
                f"Ensure depth maps exist at {self.dataset.path}/{cam}_depth/ "
                "or run fog_depthanything module first to generate them."
            ),
        )

        if depth_loaded.shape[:2] != image.shape[:2]:
            raise ValueError(
                f"Loaded depth shape {depth_loaded.shape[:2]} doesn't match "
                f"image shape {image.shape[:2]}"
            )

        depth_processed = depth_loaded * self.depth_scale

        # Clamp depth to valid range (must be > delta_z for valid reprojection)
        depth_processed = np.clip(depth_processed, 0.1, 1000.0)

        logger.debug(
            f"Frame {frame_idx} ({rgb_filename}): Processing with depth range "
            f"[{depth_processed.min():.2f}, {depth_processed.max():.2f}]m, "
            f"camera displacement {self.delta_z*100:.2f}cm"
        )

        blur_vectors, valid_mask = self._compute_blur_vectors_reprojection(
            h,
            w,
            depth_processed,
            intrinsics=intrinsics,
        )

        if self.blur_scale != 1.0:
            blur_vectors *= self.blur_scale

        result = self._apply_vector_blur(image, blur_vectors, valid_mask)

        return result

    def _compute_blur_vectors_reprojection(
        self,
        h: int,
        w: int,
        depth: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute blur vectors using proper 3D reprojection.

        Mathematical derivation:
            Given pixel (u, v) with depth Z, camera moves forward by delta_z.

            Backproject to 3D:
                X = (u - cx) * Z / fx
                Y = (v - cy) * Z / fy

            After camera motion (Z' = Z - delta_z):
                u' = fx * X / Z' + cx = (u - cx) * Z / (Z - delta_z) + cx
                v' = fy * Y / Z' + cy = (v - cy) * Z / (Z - delta_z) + cy

            Blur vector:
                delta_u = u' - u = (u - cx) * delta_z / (Z - delta_z)
                delta_v = v' - v = (v - cy) * delta_z / (Z - delta_z)

        Args:
            h: Image height
            w: Image width
            depth: Depth map (H, W) in meters (pinhole Z-depth)

        Returns:
            Tuple of:
                - Blur vectors (H, W, 2) in pixels
                - Valid mask (H, W) boolean, True where blur is valid
        """
        K = intrinsics
        fx, fy = K.fx, K.fy
        cx, cy = K.cx, K.cy

        v_coords, u_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Current depth (pinhole Z-depth along optical axis)
        Z = depth

        # Validity mask: depth must be > delta_z + epsilon for valid reprojection
        # Points with Z <= delta_z would cross behind the camera plane
        epsilon = 0.1  # 10cm safety margin
        valid_mask = (Z > self.delta_z + epsilon) & (Z > 0)

        # Backproject to 3D (camera coordinates)
        X = (u_coords - cx) * Z / fx
        Y = (v_coords - cy) * Z / fy

        # Camera moves forward by delta_z along optical axis (Z)
        # In camera frame, objects move toward camera: Z' = Z - delta_z
        X_new = X
        Y_new = Y
        Z_new = Z - self.delta_z

        # For invalid pixels (Z <= delta_z), set Z_new to safe value
        # to avoid division by zero/negative (blur will be zeroed by mask later)
        Z_new = np.where(valid_mask, Z_new, 1.0)

        # Reproject to image plane
        u_new = fx * X_new / Z_new + cx
        v_new = fy * Y_new / Z_new + cy

        # Blur vectors (displacement in pixels)
        blur_u = u_new - u_coords
        blur_v = v_new - v_coords

        blur_u = np.where(valid_mask, blur_u, 0.0)
        blur_v = np.where(valid_mask, blur_v, 0.0)

        # Clamp extreme values (prevents artifacts from depth discontinuities)
        blur_u = np.clip(blur_u, -self.max_blur_px, self.max_blur_px)
        blur_v = np.clip(blur_v, -self.max_blur_px, self.max_blur_px)

        blur_vectors = np.zeros((h, w, 2), dtype=np.float32)
        blur_vectors[:, :, 0] = blur_u
        blur_vectors[:, :, 1] = blur_v

        # Log blur statistics
        blur_magnitude = np.sqrt(blur_u**2 + blur_v**2)
        valid_blur = blur_magnitude[valid_mask]
        if len(valid_blur) > 0:
            logger.debug(
                f"Blur stats: min={valid_blur.min():.1f}px, max={valid_blur.max():.1f}px, "
                f"mean={valid_blur.mean():.1f}px, valid_pixels={valid_mask.sum()}/{valid_mask.size}"
            )

        return blur_vectors, valid_mask

    def _apply_vector_blur(
        self,
        image: np.ndarray,
        blur_vectors: np.ndarray,
        valid_mask: np.ndarray
    ) -> np.ndarray:
        """Apply motion blur using per-pixel blur vectors.

        Uses line integral convolution with adaptive sampling:
        - More samples for longer blur vectors (prevents banding)
        - Fewer samples for short blur (saves computation)
        - Triangular weighting for smoother result (optional)

        Convention: t in [-0.5, +0.5] means the input frame represents
        the mid-exposure view. Blur extends equally forward and backward in time.

        Args:
            image: Input image (H, W, 3)
            blur_vectors: Per-pixel blur vectors (H, W, 2) in pixels
            valid_mask: Boolean mask of valid pixels

        Returns:
            Blurred image
        """
        h, w = image.shape[:2]

        blur_magnitude = np.sqrt(
            blur_vectors[:, :, 0]**2 + blur_vectors[:, :, 1]**2
        )
        max_blur = np.max(blur_magnitude)

        # Adaptive sample count based on maximum blur length
        num_samples = int(np.ceil(max_blur * self.samples_per_pixel))
        num_samples = max(self.min_samples, min(self.max_samples, num_samples))

        # Ensure odd number for symmetric sampling around t=0
        if num_samples % 2 == 0:
            num_samples += 1

        logger.debug(f"Using {num_samples} samples (max blur: {max_blur:.1f} px)")

        # Base pixel coordinates
        v_coords, u_coords = np.mgrid[0:h, 0:w].astype(np.float32)

        # Accumulate samples along blur path with triangular weighting
        result = np.zeros((h, w, 3), dtype=np.float32)
        weight_sum = np.zeros((h, w, 1), dtype=np.float32)

        for i in range(num_samples):
            # Sample position along blur vector
            # t goes from -0.5 to +0.5 (frame at mid-exposure)
            t = (i / (num_samples - 1)) - 0.5

            # Triangular weighting: more weight near t=0 (mid-exposure)
            # This produces smoother blur than box filter
            weight = 1.0 - 2.0 * abs(t)  # 1 at t=0, 0 at t=+-0.5

            sample_u = u_coords + t * blur_vectors[:, :, 0]
            sample_v = v_coords + t * blur_vectors[:, :, 1]

            in_bounds = (
                (sample_u >= 0) & (sample_u < w) &
                (sample_v >= 0) & (sample_v < h)
            )

            # Clamp for remap (it will handle border, but we track validity)
            sample_u_clamped = np.clip(sample_u, 0, w - 1)
            sample_v_clamped = np.clip(sample_v, 0, h - 1)

            # Sample the image using bilinear interpolation
            sampled = cv2.remap(
                image,
                sample_u_clamped,
                sample_v_clamped,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            pixel_weight = weight * in_bounds.astype(np.float32)
            pixel_weight = pixel_weight[:, :, np.newaxis]

            result += sampled.astype(np.float32) * pixel_weight
            weight_sum += pixel_weight

        # Normalize by weight sum (avoids darkening at edges)
        weight_sum = np.maximum(weight_sum, 1e-6)  # Avoid division by zero
        result = result / weight_sum

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._intrinsics_by_camera = {}
        logger.debug("Speed blur module cleaned up")
