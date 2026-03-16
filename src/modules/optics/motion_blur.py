"""Motion blur perturbation module.

.. deprecated::
    Use :class:`SpeedBlurModule` instead.
"""

import warnings
import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


class BlurType(Enum):
    """Types of motion blur."""
    LINEAR = "linear"          # Straight-line motion
    RADIAL = "radial"          # Rotation around center
    ZOOM = "zoom"              # Forward/backward motion


@dataclass
class MotionBlurParameters:
    """Parameters for motion blur simulation."""

    blur_type: str = field(
        default="linear",
        metadata={"choices": [blur_type.value for blur_type in BlurType]},
    )
    """Type of blur: 'linear', 'radial', or 'zoom'"""

    intensity: float = 0.5
    """Blur intensity (0.0 = none, 1.0 = maximum)"""

    angle: float = 0.0
    """Blur direction in degrees (0=right, 90=down, 180=left, 270=up)"""

    kernel_size: int = 15
    """Size of blur kernel in pixels (must be odd)"""

    use_depth: bool = True
    """Use depth information for realistic blur (closer objects blur more)"""

    depth_scale: float = 1.0
    """Multiplier for depth effect (higher = more depth variation)"""

    random_variation: bool = False
    """Add random variation per frame (realistic vibrations)"""

    variation_range: float = 0.2
    """Range of random variation (0.0-1.0)"""


class MotionBlurPresets:
    """Common motion blur presets."""

    @staticmethod
    def get_preset(name: str) -> MotionBlurParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            MotionBlurParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Camera shake (slight vibration)
            "camera_shake": MotionBlurParameters(
                blur_type="linear",
                intensity=0.3,
                angle=0.0,
                kernel_size=7,
                use_depth=False,
                random_variation=True,
                variation_range=0.3
            ),

            # Fast horizontal pan (turning camera)
            "fast_turn": MotionBlurParameters(
                blur_type="linear",
                intensity=0.7,
                angle=0.0,  # Horizontal
                kernel_size=25,
                use_depth=True,
                depth_scale=0.5,
                random_variation=False
            ),

            # Forward motion (driving fast)
            "forward_motion": MotionBlurParameters(
                blur_type="zoom",
                intensity=0.6,
                angle=0.0,  # Not used for zoom
                kernel_size=21,
                use_depth=True,
                depth_scale=2.0,  # Strong depth effect
                random_variation=False
            ),

            # Severe shake (rough terrain)
            "severe_shake": MotionBlurParameters(
                blur_type="linear",
                intensity=0.8,
                angle=45.0,  # Diagonal
                kernel_size=31,
                use_depth=False,
                random_variation=True,
                variation_range=0.5
            ),

            # Rotation blur (spinning)
            "rotation": MotionBlurParameters(
                blur_type="radial",
                intensity=0.6,
                angle=0.0,  # Not used for radial
                kernel_size=21,
                use_depth=True,
                depth_scale=1.0,
                random_variation=False
            ),

            # Emergency brake (forward zoom blur)
            "emergency_brake": MotionBlurParameters(
                blur_type="zoom",
                intensity=0.9,
                angle=0.0,
                kernel_size=35,
                use_depth=True,
                depth_scale=3.0,  # Very strong depth effect
                random_variation=False
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown motion blur preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class MotionBlurModule(PerturbationModule):
    """Motion blur perturbation.

    Simulates camera motion blur from movement, vibrations, or fast motion.
    Supports depth-aware blur for realistic effects.
    """

    module_name = "motion_blur"
    module_description = "[DEPRECATED] Motion blur from camera movement - use speed_blur instead"

    # Deprecation settings
    deprecated = True
    deprecation_message = "Use speed_blur for physically-accurate depth-dependent motion blur"
    replacement = "speed_blur"

    PARAMETERS_CLASS = MotionBlurParameters

    def _setup(self, context) -> None:
        """Setup motion blur module."""
        warnings.warn(
            "MotionBlurModule is deprecated and will be removed in a future version. "
            "Use SpeedBlurModule instead, which provides physically-accurate "
            "depth-dependent motion blur using 3D reprojection.",
            DeprecationWarning,
            stacklevel=2
        )
        logger.warning(
            "MotionBlurModule is deprecated. Use SpeedBlurModule for "
            "physically-accurate depth-dependent motion blur."
        )
        params = self.parameters or {}

        if 'preset' in params:
            preset_name = params['preset']
            preset_params = MotionBlurPresets.get_preset(preset_name)

            self.blur_type = preset_params.blur_type
            self.intensity = preset_params.intensity
            self.angle = preset_params.angle
            self.kernel_size = preset_params.kernel_size
            self.use_depth = preset_params.use_depth
            self.depth_scale = preset_params.depth_scale
            self.random_variation = preset_params.random_variation
            self.variation_range = preset_params.variation_range

            # Allow preset override
            self.blur_type = params.get('blur_type', self.blur_type)
            self.intensity = params.get('intensity', self.intensity)
            self.angle = params.get('angle', self.angle)
            self.kernel_size = params.get('kernel_size', self.kernel_size)
            self.use_depth = params.get('use_depth', self.use_depth)
            self.depth_scale = params.get('depth_scale', self.depth_scale)
            self.random_variation = params.get('random_variation', self.random_variation)
            self.variation_range = params.get('variation_range', self.variation_range)

        else:
            # No preset - use explicit parameters or defaults
            self.blur_type = params.get('blur_type', 'linear')
            self.intensity = params.get('intensity', 0.5)
            self.angle = params.get('angle', 0.0)
            self.kernel_size = params.get('kernel_size', 15)
            self.use_depth = params.get('use_depth', True)
            self.depth_scale = params.get('depth_scale', 1.0)
            self.random_variation = params.get('random_variation', False)
            self.variation_range = params.get('variation_range', 0.2)

        # Ensure kernel size is odd
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        logger.info(f"Motion blur module initialized:")
        logger.info(f"  Type: {self.blur_type}")
        logger.info(f"  Intensity: {self.intensity}")
        logger.info(f"  Kernel size: {self.kernel_size}")
        logger.info(f"  Depth-aware: {self.use_depth}")
        if self.blur_type == 'linear':
            logger.info(f"  Angle: {self.angle}°")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply motion blur to image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            depth: Depth map (H, W) for depth-aware blur
            frame_idx: Frame index
            **kwargs: Additional context (unused)

        Returns:
            Image with motion blur applied
        """
        intensity = self.intensity
        angle = self.angle

        if self.random_variation:
            # Random variation in intensity
            variation = np.random.uniform(-self.variation_range, self.variation_range)
            intensity = np.clip(self.intensity + variation, 0.0, 1.0)

            # Random variation in angle (for camera shake)
            if self.blur_type == 'linear':
                angle = self.angle + np.random.uniform(-30, 30)

        if self.blur_type == 'linear':
            blurred = self._apply_linear_blur(image, depth, intensity, angle)
        elif self.blur_type == 'radial':
            blurred = self._apply_radial_blur(image, depth, intensity)
        elif self.blur_type == 'zoom':
            blurred = self._apply_zoom_blur(image, depth, intensity)
        else:
            logger.warning(f"Unknown blur type: {self.blur_type}, skipping")
            return image

        return blurred

    def _apply_linear_blur(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        intensity: float,
        angle: float
    ) -> np.ndarray:
        """Apply directional linear motion blur.

        Args:
            image: Input image
            depth: Depth map (optional)
            intensity: Blur intensity
            angle: Blur direction in degrees

        Returns:
            Blurred image
        """
        h, w = image.shape[:2]

        # Calculate kernel size based on intensity
        kernel_size = int(self.kernel_size * intensity)
        if kernel_size < 3:
            return image
        if kernel_size % 2 == 0:
            kernel_size += 1

        # If depth-aware and depth is available
        if self.use_depth and depth is not None:
            return self._apply_depth_aware_blur(image, depth, intensity, angle, kernel_size)

        # Simple uniform blur (no depth)
        kernel = self._create_motion_kernel(kernel_size, angle)
        blurred = cv2.filter2D(image, -1, kernel)

        return blurred

    def _apply_depth_aware_blur(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        intensity: float,
        angle: float,
        kernel_size: int
    ) -> np.ndarray:
        """Apply depth-aware motion blur (closer objects blur more).

        Args:
            image: Input image
            depth: Depth map
            intensity: Blur intensity
            angle: Blur direction
            kernel_size: Base kernel size

        Returns:
            Depth-aware blurred image
        """
        h, w = image.shape[:2]

        # Normalize depth to [0, 1] (0 = close, 1 = far)
        depth_norm = depth.astype(np.float32)
        if depth_norm.max() > 0:
            depth_norm = depth_norm / depth_norm.max()

        # Invert: 1 = close, 0 = far
        depth_inv = 1.0 - depth_norm

        depth_inv = depth_inv * self.depth_scale
        depth_inv = np.clip(depth_inv, 0, 1)

        num_levels = 5
        blurred_levels = []

        for i in range(num_levels):
            level_kernel_size = int(kernel_size * (i + 1) / num_levels)
            if level_kernel_size < 3:
                level_kernel_size = 3
            if level_kernel_size % 2 == 0:
                level_kernel_size += 1

            kernel = self._create_motion_kernel(level_kernel_size, angle)
            blurred = cv2.filter2D(image, -1, kernel)
            blurred_levels.append(blurred)

        result = np.zeros_like(image, dtype=np.float32)

        for i in range(num_levels):
            level_start = i / num_levels
            level_end = (i + 1) / num_levels

            weight = np.zeros_like(depth_inv)
            mask = (depth_inv >= level_start) & (depth_inv < level_end)
            weight[mask] = 1.0

            weight = cv2.GaussianBlur(weight, (11, 11), 3.0)
            weight = np.clip(weight, 0, 1)

            weight_3ch = weight[:, :, np.newaxis]
            result += blurred_levels[i].astype(np.float32) * weight_3ch

        # Normalize
        total_weight = np.sum([cv2.GaussianBlur((depth_inv >= i/num_levels).astype(np.float32), (11, 11), 3.0)
                               for i in range(num_levels)], axis=0)
        total_weight = np.maximum(total_weight, 1e-6)
        total_weight_3ch = total_weight[:, :, np.newaxis]

        result = result / total_weight_3ch
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _apply_radial_blur(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        intensity: float
    ) -> np.ndarray:
        """Apply radial motion blur (rotation effect).

        Args:
            image: Input image
            depth: Depth map (optional)
            intensity: Blur intensity

        Returns:
            Radially blurred image
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Number of rotations to blend
        num_rotations = int(10 * intensity)
        if num_rotations < 2:
            return image

        # Accumulate rotated versions
        result = image.astype(np.float32)

        for i in range(1, num_rotations):
            # Small rotation angle
            angle = (i / num_rotations) * 3.0 * intensity  # Up to 3 degrees per rotation

            # Rotation matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

            result += rotated.astype(np.float32)

        result = result / num_rotations
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _apply_zoom_blur(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        intensity: float
    ) -> np.ndarray:
        """Apply zoom motion blur (forward/backward motion).

        Args:
            image: Input image
            depth: Depth map (optional, enhances effect)
            intensity: Blur intensity

        Returns:
            Zoom-blurred image
        """
        h, w = image.shape[:2]
        center_x, center_y = w // 2, h // 2

        # Number of zoom steps
        num_steps = int(10 * intensity)
        if num_steps < 2:
            return image

        # Depth weight if available
        depth_weight = None
        if self.use_depth and depth is not None:
            # Normalize depth
            depth_norm = depth.astype(np.float32)
            if depth_norm.max() > 0:
                depth_norm = depth_norm / depth_norm.max()
            # Closer objects blur more
            depth_weight = (1.0 - depth_norm) * self.depth_scale
            depth_weight = np.clip(depth_weight, 0, 1)

        # Accumulate zoomed versions
        result = image.astype(np.float32)

        for i in range(1, num_steps):
            # Zoom factor (1.0 to 1.0 + intensity*0.1)
            scale = 1.0 + (i / num_steps) * intensity * 0.1

            # Scale matrix
            M = cv2.getRotationMatrix2D((center_x, center_y), 0, scale)
            zoomed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

            if depth_weight is not None:
                # Weight by depth (closer = more blur)
                weight = depth_weight[:, :, np.newaxis]
                result += zoomed.astype(np.float32) * weight
            else:
                result += zoomed.astype(np.float32)

        if depth_weight is not None:
            # Normalize by total weight
            total_weight = 1.0 + np.sum(depth_weight) * (num_steps - 1) / (h * w)
            result = result / total_weight
        else:
            result = result / num_steps

        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _create_motion_kernel(self, size: int, angle: float) -> np.ndarray:
        """Create motion blur kernel.

        Args:
            size: Kernel size (odd number)
            angle: Motion direction in degrees

        Returns:
            Motion blur kernel
        """
        kernel = np.zeros((size, size))

        # Center point
        center = size // 2

        angle_rad = np.deg2rad(angle)

        # Draw line in the direction of motion
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)

        for i in range(size):
            offset = i - center
            x = int(center + offset * cos_a)
            y = int(center + offset * sin_a)

            if 0 <= x < size and 0 <= y < size:
                kernel[y, x] = 1.0

        # Normalize
        kernel = kernel / np.sum(kernel)

        return kernel

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Motion blur module cleaned up")
