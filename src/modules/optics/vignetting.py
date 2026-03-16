"""Lens vignetting / edge darkening perturbation."""

import numpy as np
from typing import Optional
from dataclasses import dataclass

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class VignetteParameters:
    """Parameters for vignetting simulation."""

    intensity: float = 0.5
    """Vignette intensity (0-1), where 0=no vignette, 1=maximum darkening at edges"""

    radius: float = 0.8
    """Vignette radius as fraction of image diagonal (0-1), where vignetting starts"""

    falloff: float = 4.0
    """Falloff exponent for cos^n law (4.0 = physically accurate cos^4 vignetting)"""

    center_x: float = 0.5
    """Center X position as fraction of image width (0.5 = centered)"""

    center_y: float = 0.5
    """Center Y position as fraction of image height (0.5 = centered)"""


class VignetteModule(PerturbationModule):
    """Lens vignetting / edge darkening perturbation."""

    module_name = "vignetting"
    module_description = "Lens vignetting/edge darkening simulation"

    PARAMETERS_CLASS = VignetteParameters

    def __init__(self, config):
        """Initialize vignetting module.

        Args:
            config: PerturbationConfig with vignetting parameters
        """
        from ...config.schema import PerturbationConfig
        super().__init__(config)

        # Parse parameters from config
        parameters = self.parameters if self.parameters else {}

        self.params = VignetteParameters(
            intensity=parameters.get('intensity', 0.5),
            radius=parameters.get('radius', 0.8),
            falloff=parameters.get('falloff', 4.0),
            center_x=parameters.get('center_x', 0.5),
            center_y=parameters.get('center_y', 0.5)
        )

        # Cache for vignette mask (will be created on first apply)
        self._vignette_mask = None
        self._cached_shape = None

    def _setup(self, context) -> None:
        """Initialize vignetting module."""
        logger.info(f"  Intensity: {self.params.intensity}")
        logger.info(f"  Radius: {self.params.radius}")
        logger.info(f"  Falloff: {self.params.falloff}")
        logger.info(f"  Center: ({self.params.center_x}, {self.params.center_y})")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply vignetting effect to image.

        Args:
            image: Input image (H, W, 3) in range [0, 255], uint8
            depth: Depth map (unused for vignetting)
            frame_idx: Frame index (unused for static vignetting)
            **kwargs: Additional context (unused)

        Returns:
            Vignetted image (H, W, 3) in range [0, 255], uint8
        """
        h, w = image.shape[:2]

        if self._vignette_mask is None or self._cached_shape != (h, w):
            self._vignette_mask = self._create_vignette_mask(h, w)
            self._cached_shape = (h, w)

        img_float = image.astype(np.float32)

        # Expand mask to match image channels
        vignette_3ch = np.expand_dims(self._vignette_mask, axis=2)
        img_float = img_float * vignette_3ch

        # Clamp to valid range and convert back to uint8
        img_float = np.clip(img_float, 0, 255)
        return img_float.astype(np.uint8)

    def _create_vignette_mask(self, h: int, w: int) -> np.ndarray:
        """Create a radial vignette mask using a configurable cosine falloff."""
        # Calculate center point in pixels
        center_x = w * self.params.center_x
        center_y = h * self.params.center_y

        y, x = np.ogrid[:h, :w]

        # Calculate distance from center in pixels
        dist_x = x - center_x
        dist_y = y - center_y
        distance = np.sqrt(dist_x**2 + dist_y**2)

        # Smaller radius produces stronger edge falloff.
        diagonal = np.sqrt(h**2 + w**2) / 2
        focal_length = diagonal / (self.params.radius * np.tan(np.pi / 4))

        theta = np.arctan(distance / focal_length)

        cos_theta = np.cos(theta)
        vignette = cos_theta ** self.params.falloff

        mask = 1.0 - self.params.intensity * (1.0 - vignette)

        return mask.astype(np.float32)

    def _cleanup(self) -> None:
        """Clean up vignetting module resources."""
        logger.debug(f"Cleaning up {self.name} module")
        self._vignette_mask = None
        self._cached_shape = None
