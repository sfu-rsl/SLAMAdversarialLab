"""Lens patch / fixed occlusion perturbation."""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


class PatchShape(Enum):
    """Supported patch shapes."""
    RECTANGLE = "rectangle"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"


class PatchPosition(Enum):
    """Predefined patch positions."""
    CENTER = "center"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    LEFT_EDGE = "left_edge"
    RIGHT_EDGE = "right_edge"
    TOP_EDGE = "top_edge"
    BOTTOM_EDGE = "bottom_edge"
    CUSTOM = "custom"


@dataclass
class LensPatchParameters:
    """Parameters for lens patch simulation."""

    shape: str = field(
        default="circle",
        metadata={"choices": [shape.value for shape in PatchShape]},
    )
    """Patch shape: 'circle', 'rectangle', or 'ellipse'"""

    position: str = field(
        default="bottom_right",
        metadata={"choices": [position.value for position in PatchPosition]},
    )
    """Patch position: predefined location or 'custom' with center_x/center_y"""

    size_percent: float = 15.0
    """Patch size as percentage of image width (10-50)"""

    aspect_ratio: float = 1.0
    """Aspect ratio for ellipse/rectangle (width/height)"""

    center_x: Optional[float] = None
    """Custom X position as fraction of width (0-1), only if position='custom'"""

    center_y: Optional[float] = None
    """Custom Y position as fraction of height (0-1), only if position='custom'"""

    opacity: float = 1.0
    """Patch opacity: 0 (transparent) to 1 (fully opaque)"""

    color: Tuple[int, int, int] = (0, 0, 0)
    """Patch color in RGB (0-255)"""

    blur_edge: bool = True
    """Apply Gaussian blur to patch edges for realistic effect"""

    blur_sigma: float = 5.0
    """Sigma for Gaussian blur of patch edges"""


class LensPatchPresets:
    """Common lens patch presets."""

    @staticmethod
    def get_preset(name: str) -> LensPatchParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            LensPatchParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Small patches
            "small_corner": LensPatchParameters(
                shape="circle",
                position="bottom_right",
                size_percent=10.0,
                opacity=1.0,
                color=(0, 0, 0)
            ),
            "small_center": LensPatchParameters(
                shape="circle",
                position="center",
                size_percent=15.0,
                opacity=0.7,
                color=(50, 50, 50)  # Gray
            ),

            # Medium patches
            "medium_dirt": LensPatchParameters(
                shape="ellipse",
                position="top_left",
                size_percent=20.0,
                aspect_ratio=1.5,
                opacity=0.6,
                color=(80, 70, 60)  # Brown-ish
            ),
            "medium_tape": LensPatchParameters(
                shape="rectangle",
                position="left_edge",
                size_percent=25.0,
                aspect_ratio=0.3,  # Tall and thin
                opacity=0.9,
                color=(30, 30, 30)
            ),

            # Large patches
            "large_obstruction": LensPatchParameters(
                shape="circle",
                position="center",
                size_percent=30.0,
                opacity=1.0,
                color=(0, 0, 0)
            ),
            "large_damage": LensPatchParameters(
                shape="ellipse",
                position="bottom_left",
                size_percent=35.0,
                aspect_ratio=1.2,
                opacity=0.8,
                color=(20, 20, 20)
            ),

            # Edge occlusions
            "edge_left": LensPatchParameters(
                shape="rectangle",
                position="left_edge",
                size_percent=40.0,
                aspect_ratio=0.15,
                opacity=1.0,
                color=(0, 0, 0)
            ),
            "edge_bottom": LensPatchParameters(
                shape="rectangle",
                position="bottom_edge",
                size_percent=40.0,
                aspect_ratio=5.0,  # Wide and short
                opacity=1.0,
                color=(0, 0, 0)
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown lens patch preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class LensPatchModule(PerturbationModule):
    """Lens patch / fixed occlusion perturbation."""

    module_name = "lens_patch"
    module_description = "Camera lens patch/occlusion simulation"

    PARAMETERS_CLASS = LensPatchParameters

    def _setup(self, context) -> None:
        """Setup lens patch module."""
        params = self.parameters or {}

        if 'preset' in params:
            preset_name = params['preset']
            preset_params = LensPatchPresets.get_preset(preset_name)

            self.shape = preset_params.shape
            self.position = preset_params.position
            self.size_percent = preset_params.size_percent
            self.aspect_ratio = preset_params.aspect_ratio
            self.center_x = preset_params.center_x
            self.center_y = preset_params.center_y
            self.opacity = preset_params.opacity
            self.color = preset_params.color
            self.blur_edge = preset_params.blur_edge
            self.blur_sigma = preset_params.blur_sigma

            # Allow preset override with explicit parameters
            self.shape = params.get('shape', self.shape)
            self.position = params.get('position', self.position)
            self.size_percent = params.get('size_percent', self.size_percent)
            self.aspect_ratio = params.get('aspect_ratio', self.aspect_ratio)
            self.opacity = params.get('opacity', self.opacity)
            self.color = tuple(params.get('color', self.color))
            self.blur_edge = params.get('blur_edge', self.blur_edge)
            self.blur_sigma = params.get('blur_sigma', self.blur_sigma)

        else:
            # No preset - use explicit parameters or defaults
            self.shape = params.get('shape', 'circle')
            self.position = params.get('position', 'bottom_right')
            self.size_percent = params.get('size_percent', 15.0)
            self.aspect_ratio = params.get('aspect_ratio', 1.0)
            self.center_x = params.get('center_x', None)
            self.center_y = params.get('center_y', None)
            self.opacity = params.get('opacity', 1.0)
            self.color = tuple(params.get('color', [0, 0, 0]))
            self.blur_edge = params.get('blur_edge', True)
            self.blur_sigma = params.get('blur_sigma', 5.0)

        if self.position == "custom" and (self.center_x is None or self.center_y is None):
            raise ValueError("Custom position requires 'center_x' and 'center_y' parameters")

        # Pre-compute patch mask (will be computed on first frame when we know image size)
        self.patch_mask = None
        self.image_shape = None

        logger.info(f"Lens patch module initialized:")
        logger.info(f"  Shape: {self.shape}")
        logger.info(f"  Position: {self.position}")
        logger.info(f"  Size: {self.size_percent}% of width")
        logger.info(f"  Opacity: {self.opacity}")
        logger.info(f"  Color: RGB{self.color}")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply lens patch to image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            depth: Depth map (unused for lens patch)
            frame_idx: Frame index (unused - patch is static)
            **kwargs: Additional context (unused)

        Returns:
            Image with lens patch applied
        """
        # Generate mask on first frame (reuse for all subsequent frames)
        if self.patch_mask is None or self.image_shape != image.shape[:2]:
            self.image_shape = image.shape[:2]
            self.patch_mask = self._generate_patch_mask(image.shape[:2])

        result = image.copy()

        patch_overlay = np.full_like(image, self.color, dtype=np.uint8)

        alpha = self.patch_mask[:, :, np.newaxis]
        result = (image * (1 - alpha) + patch_overlay * alpha).astype(np.uint8)

        return result

    def _generate_patch_mask(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """Generate patch mask for the given image shape.

        Args:
            image_shape: Image shape (height, width)

        Returns:
            Mask array (H, W) with values from 0 (no patch) to opacity (full patch)
        """
        h, w = image_shape

        patch_radius_px = int((self.size_percent / 100.0) * w / 2)

        if self.position == "custom":
            cx = int(self.center_x * w)
            cy = int(self.center_y * h)
        else:
            cx, cy = self._get_position_coordinates(w, h, patch_radius_px)

        if self.shape == "circle":
            mask = self._create_circle_mask(h, w, cx, cy, patch_radius_px)
        elif self.shape == "rectangle":
            mask = self._create_rectangle_mask(h, w, cx, cy, patch_radius_px)
        elif self.shape == "ellipse":
            mask = self._create_ellipse_mask(h, w, cx, cy, patch_radius_px)
        else:
            raise ValueError(f"Unknown shape: {self.shape}")

        mask = mask * self.opacity

        if self.blur_edge and self.blur_sigma > 0:
            try:
                import cv2
                kernel_size = int(self.blur_sigma * 6) | 1  # Ensure odd
                mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), self.blur_sigma)
            except ImportError:
                logger.warning("OpenCV not available, skipping edge blur")

        return mask

    def _get_position_coordinates(
        self,
        width: int,
        height: int,
        patch_radius: int
    ) -> Tuple[int, int]:
        """Get patch center coordinates based on position name.

        Args:
            width: Image width
            height: Image height
            patch_radius: Patch radius in pixels

        Returns:
            (cx, cy) center coordinates
        """
        margin = patch_radius  # Keep patch within image bounds

        positions = {
            "center": (width // 2, height // 2),
            "top_left": (margin, margin),
            "top_right": (width - margin, margin),
            "bottom_left": (margin, height - margin),
            "bottom_right": (width - margin, height - margin),
            "left_edge": (margin, height // 2),
            "right_edge": (width - margin, height // 2),
            "top_edge": (width // 2, margin),
            "bottom_edge": (width // 2, height - margin),
        }

        if self.position not in positions:
            raise ValueError(f"Unknown position: {self.position}")

        return positions[self.position]

    def _create_circle_mask(
        self,
        h: int,
        w: int,
        cx: int,
        cy: int,
        radius: int
    ) -> np.ndarray:
        """Create circular patch mask.

        Args:
            h: Image height
            w: Image width
            cx: Center X coordinate
            cy: Center Y coordinate
            radius: Circle radius

        Returns:
            Binary mask (H, W) with 1.0 inside circle, 0.0 outside
        """
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
        mask = (dist_from_center <= radius).astype(np.float32)
        return mask

    def _create_rectangle_mask(
        self,
        h: int,
        w: int,
        cx: int,
        cy: int,
        base_size: int
    ) -> np.ndarray:
        """Create rectangular patch mask.

        Args:
            h: Image height
            w: Image width
            cx: Center X coordinate
            cy: Center Y coordinate
            base_size: Base size (will be modified by aspect ratio)

        Returns:
            Binary mask (H, W) with 1.0 inside rectangle, 0.0 outside
        """
        width_half = int(base_size * np.sqrt(self.aspect_ratio))
        height_half = int(base_size / np.sqrt(self.aspect_ratio))

        mask = np.zeros((h, w), dtype=np.float32)

        # Calculate bounds
        x1 = max(0, cx - width_half)
        x2 = min(w, cx + width_half)
        y1 = max(0, cy - height_half)
        y2 = min(h, cy + height_half)

        mask[y1:y2, x1:x2] = 1.0
        return mask

    def _create_ellipse_mask(
        self,
        h: int,
        w: int,
        cx: int,
        cy: int,
        base_radius: int
    ) -> np.ndarray:
        """Create elliptical patch mask.

        Args:
            h: Image height
            w: Image width
            cx: Center X coordinate
            cy: Center Y coordinate
            base_radius: Base radius (will be modified by aspect ratio)

        Returns:
            Binary mask (H, W) with 1.0 inside ellipse, 0.0 outside
        """
        radius_x = int(base_radius * np.sqrt(self.aspect_ratio))
        radius_y = int(base_radius / np.sqrt(self.aspect_ratio))

        y, x = np.ogrid[:h, :w]
        mask = (((x - cx) / radius_x)**2 + ((y - cy) / radius_y)**2 <= 1).astype(np.float32)
        return mask

    def _cleanup(self) -> None:
        """Clean up resources."""
        self.patch_mask = None
        logger.debug("Lens patch module cleaned up")
