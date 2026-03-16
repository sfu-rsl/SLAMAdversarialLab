"""Lens flare perturbation from bright light sources."""

import numpy as np
import cv2
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


class FlareType(Enum):
    """Types of lens flare patterns."""
    SUBTLE = "subtle"
    MODERATE = "moderate"
    STRONG = "strong"
    EXTREME = "extreme"


@dataclass
class LensFlareParameters:
    """Parameters for lens flare simulation."""

    intensity: float = 0.6
    """Flare intensity (0.0 to 1.0)"""

    sun_position: Optional[Tuple[float, float]] = None
    """Sun position as fraction of image (x, y) in [0, 1]. If None, auto-detect bright regions."""

    num_ghosts: int = 5
    """Number of ghost reflections (halo artifacts)"""

    add_streaks: bool = True
    """Add light streaks/rays"""

    num_streaks: int = 6
    """Number of light streaks (usually 6 for hexagonal aperture)"""

    streak_length: float = 0.4
    """Length of streaks as fraction of image diagonal"""

    add_glow: bool = True
    """Add glow around the bright source"""

    glow_radius: float = 0.15
    """Glow radius as fraction of image width"""

    color_tint: Tuple[float, float, float] = (1.0, 0.95, 0.85)
    """Color tint for flare (R, G, B) - warm/yellowish by default"""

    chromatic_aberration: bool = True
    """Add chromatic aberration (color fringing)"""


class LensFlarePresets:
    """Common lens flare presets."""

    @staticmethod
    def get_preset(name: str) -> LensFlareParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            LensFlareParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Strong flare - significant artifacts
            "strong": LensFlareParameters(
                intensity=0.7,
                num_ghosts=7,
                add_streaks=True,
                num_streaks=6,
                streak_length=0.5,
                add_glow=True,
                glow_radius=0.18,
                chromatic_aberration=True
            ),

            # Extreme flare - severe degradation
            "extreme": LensFlareParameters(
                intensity=0.9,
                num_ghosts=10,
                add_streaks=True,
                num_streaks=8,
                streak_length=0.7,
                add_glow=True,
                glow_radius=0.25,
                chromatic_aberration=True
            ),

            # Sun in top corner (common driving scenario)
            "top_corner_sun": LensFlareParameters(
                intensity=0.6,
                sun_position=(0.85, 0.15),  # Top-right
                num_ghosts=6,
                add_streaks=True,
                num_streaks=6,
                streak_length=0.4,
                add_glow=True,
                glow_radius=0.15,
                chromatic_aberration=True
            ),

            # Overhead sun (top-center)
            "overhead_sun": LensFlareParameters(
                intensity=0.65,
                sun_position=(0.5, 0.15),  # Top-center
                num_ghosts=7,
                add_streaks=True,
                num_streaks=6,
                streak_length=0.5,
                add_glow=True,
                glow_radius=0.18,
                chromatic_aberration=True
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown lens flare preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class LensFlareModule(PerturbationModule):
    """Lens flare perturbation from bright light sources."""

    module_name = "lens_flare"
    module_description = "Lens flare from bright light sources"

    PARAMETERS_CLASS = LensFlareParameters

    def _setup(self, context) -> None:
        """Setup lens flare module."""
        params = self.parameters or {}

        if 'preset' in params:
            preset_name = params['preset']
            preset_params = LensFlarePresets.get_preset(preset_name)

            self.intensity = preset_params.intensity
            self.sun_position = preset_params.sun_position
            self.num_ghosts = preset_params.num_ghosts
            self.add_streaks = preset_params.add_streaks
            self.num_streaks = preset_params.num_streaks
            self.streak_length = preset_params.streak_length
            self.add_glow = preset_params.add_glow
            self.glow_radius = preset_params.glow_radius
            self.color_tint = preset_params.color_tint
            self.chromatic_aberration = preset_params.chromatic_aberration

            self.intensity = params.get('intensity', self.intensity)
            self.sun_position = params.get('sun_position', self.sun_position)
            self.num_ghosts = params.get('num_ghosts', self.num_ghosts)
            self.add_streaks = params.get('add_streaks', self.add_streaks)
            self.num_streaks = params.get('num_streaks', self.num_streaks)
            self.streak_length = params.get('streak_length', self.streak_length)
            self.add_glow = params.get('add_glow', self.add_glow)
            self.glow_radius = params.get('glow_radius', self.glow_radius)
            self.color_tint = tuple(params.get('color_tint', self.color_tint))
            self.chromatic_aberration = params.get('chromatic_aberration', self.chromatic_aberration)

        else:
            # No preset - use explicit parameters or defaults
            self.intensity = params.get('intensity', 0.6)
            self.sun_position = params.get('sun_position', None)
            self.num_ghosts = params.get('num_ghosts', 5)
            self.add_streaks = params.get('add_streaks', True)
            self.num_streaks = params.get('num_streaks', 6)
            self.streak_length = params.get('streak_length', 0.4)
            self.add_glow = params.get('add_glow', True)
            self.glow_radius = params.get('glow_radius', 0.15)
            self.color_tint = tuple(params.get('color_tint', [1.0, 0.95, 0.85]))
            self.chromatic_aberration = params.get('chromatic_aberration', True)
        if self.sun_position is not None and isinstance(self.sun_position, list):
            self.sun_position = tuple(self.sun_position)

        logger.info(f"Lens flare module initialized:")
        logger.info(f"  Intensity: {self.intensity}")
        logger.info(f"  Sun position: {self.sun_position or 'auto-detect'}")
        logger.info(f"  Ghosts: {self.num_ghosts}")
        logger.info(f"  Streaks: {self.num_streaks if self.add_streaks else 'disabled'}")
        logger.info(f"  Glow: {self.glow_radius if self.add_glow else 'disabled'}")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply lens flare to image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            depth: Depth map (unused for lens flare)
            frame_idx: Frame index
            **kwargs: Additional context (unused)

        Returns:
            Image with lens flare applied
        """
        h, w = image.shape[:2]

        if self.sun_position is None:
            sun_x, sun_y = self._detect_bright_region(image)
        else:
            sun_x = int(self.sun_position[0] * w)
            sun_y = int(self.sun_position[1] * h)

        visibility = self._check_visibility(image, sun_x, sun_y, w, h)

        if visibility < 0.1:
            return image

        flare = np.zeros((h, w, 3), dtype=np.float32)

        if self.add_glow:
            self._add_glow(flare, sun_x, sun_y, w, h)

        self._add_ghosts(flare, sun_x, sun_y, w, h)

        if self.add_streaks:
            self._add_streaks(flare, sun_x, sun_y, w, h)

        if self.chromatic_aberration:
            flare = self._add_chromatic_aberration(flare, sun_x, sun_y, w, h)

        flare[:, :, 0] *= self.color_tint[0]
        flare[:, :, 1] *= self.color_tint[1]
        flare[:, :, 2] *= self.color_tint[2]

        flare *= self.intensity * visibility

        # Blend with original image (screen blend mode for realistic flare)
        result = image.astype(np.float32) / 255.0
        flare_normalized = np.clip(flare, 0, 1)

        # Screen blend: 1 - (1 - a) * (1 - b)
        result = 1.0 - (1.0 - result) * (1.0 - flare_normalized)
        result = np.clip(result * 255, 0, 255).astype(np.uint8)

        return result

    def _detect_bright_region(self, image: np.ndarray) -> Tuple[int, int]:
        """Auto-detect brightest region in image (likely sun position).

        Args:
            image: Input image

        Returns:
            (x, y) coordinates of brightest region
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(gray)

        return max_loc  # (x, y)

    def _check_visibility(
        self,
        image: np.ndarray,
        sun_x: int,
        sun_y: int,
        w: int,
        h: int
    ) -> float:
        """Check if sun position is visible or occluded by objects.

        Args:
            image: Input image
            sun_x, sun_y: Sun position
            w, h: Image dimensions

        Returns:
            Visibility factor from 0 (completely occluded) to 1 (fully visible)
        """
        # Sample region around sun position
        sample_radius = max(20, int(w * 0.03))  # 3% of image width or 20px

        # Ensure bounds
        x1 = max(0, sun_x - sample_radius)
        x2 = min(w, sun_x + sample_radius)
        y1 = max(0, sun_y - sample_radius)
        y2 = min(h, sun_y + sample_radius)

        # Extract region
        region = image[y1:y2, x1:x2]

        gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
        avg_brightness = np.mean(gray_region)

        brightness_threshold_low = 80   # Below this = completely occluded
        brightness_threshold_high = 180  # Above this = fully visible

        if avg_brightness < brightness_threshold_low:
            visibility = 0.0
        elif avg_brightness > brightness_threshold_high:
            # Bright region - sun is visible
            visibility = 1.0
        else:
            # Partial occlusion - linear interpolation
            visibility = (avg_brightness - brightness_threshold_low) / \
                        (brightness_threshold_high - brightness_threshold_low)

        # Also check for very bright spots (direct sun)
        max_brightness = np.max(gray_region)
        if max_brightness > 230:  # Very bright spot detected
            visibility = max(visibility, 0.8)  # Boost visibility

        return visibility

    def _add_glow(self, flare: np.ndarray, cx: int, cy: int, w: int, h: int) -> None:
        """Add glow around the bright source.

        Args:
            flare: Flare overlay to modify
            cx, cy: Center position
            w, h: Image dimensions
        """
        radius_px = int(self.glow_radius * w)

        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)

        glow_mask = np.exp(-(dist**2) / (2 * (radius_px / 2)**2))
        glow_mask = np.clip(glow_mask, 0, 1)

        flare[:, :, 0] += glow_mask * 0.9
        flare[:, :, 1] += glow_mask * 0.85
        flare[:, :, 2] += glow_mask * 0.7

    def _add_ghosts(self, flare: np.ndarray, sun_x: int, sun_y: int, w: int, h: int) -> None:
        """Add ghost reflections (halos) along flare line.

        Args:
            flare: Flare overlay to modify
            sun_x, sun_y: Sun position
            w, h: Image dimensions
        """
        # Center of image
        cx = w // 2
        cy = h // 2

        # Vector from sun to center
        dx = cx - sun_x
        dy = cy - sun_y

        # Place ghosts along this line
        for i in range(1, self.num_ghosts + 1):
            # Position along the flare line
            t = i / (self.num_ghosts + 1)
            ghost_x = int(sun_x + dx * t * 1.5)  # Extend beyond center
            ghost_y = int(sun_y + dy * t * 1.5)

            # Skip if outside image
            if ghost_x < 0 or ghost_x >= w or ghost_y < 0 or ghost_y >= h:
                continue

            ghost_radius = int(30 * (1.0 - t * 0.5))

            y, x = np.ogrid[:h, :w]
            dist = np.sqrt((x - ghost_x)**2 + (y - ghost_y)**2)
            ghost_mask = np.maximum(0, 1.0 - dist / ghost_radius)

            ghost_intensity = 0.4 * (1.0 - t * 0.7)

            flare[:, :, 0] += ghost_mask * ghost_intensity * 0.9
            flare[:, :, 1] += ghost_mask * ghost_intensity * 0.95
            flare[:, :, 2] += ghost_mask * ghost_intensity * 1.0

    def _add_streaks(self, flare: np.ndarray, sun_x: int, sun_y: int, w: int, h: int) -> None:
        """Add light streaks/rays emanating from sun.

        Args:
            flare: Flare overlay to modify
            sun_x, sun_y: Sun position
            w, h: Image dimensions
        """
        diag = np.sqrt(w**2 + h**2)
        streak_len = int(diag * self.streak_length)

        for i in range(self.num_streaks):
            # Angle for this streak
            angle = (i * 2 * np.pi / self.num_streaks) + np.pi / 4

            # End point of streak
            end_x = int(sun_x + streak_len * np.cos(angle))
            end_y = int(sun_y + streak_len * np.sin(angle))

            # Draw line with varying thickness and opacity
            for thickness in range(1, 4):
                opacity = 0.15 * (4 - thickness) / 3

                streak_img = np.zeros((h, w), dtype=np.float32)
                cv2.line(streak_img, (sun_x, sun_y), (end_x, end_y),
                        opacity, thickness=thickness, lineType=cv2.LINE_AA)

                flare[:, :, 0] += streak_img
                flare[:, :, 1] += streak_img * 0.95
                flare[:, :, 2] += streak_img * 0.9

    def _add_chromatic_aberration(
        self,
        flare: np.ndarray,
        sun_x: int,
        sun_y: int,
        w: int,
        h: int
    ) -> np.ndarray:
        """Add chromatic aberration (color fringing).

        Args:
            flare: Flare overlay
            sun_x, sun_y: Sun position
            w, h: Image dimensions

        Returns:
            Flare with chromatic aberration applied
        """
        flare_ca = flare.copy()

        y, x = np.mgrid[:h, :w]
        dx = x - sun_x
        dy = y - sun_y

        # Small radial shift (different for each channel)
        shift_r = 2
        shift_b = -2

        # Shift red channel outward
        x_r = np.clip(x + dx * shift_r / w, 0, w - 1).astype(np.int32)
        y_r = np.clip(y + dy * shift_r / h, 0, h - 1).astype(np.int32)
        flare_ca[:, :, 0] = flare[y_r, x_r, 0]

        # Shift blue channel inward
        x_b = np.clip(x + dx * shift_b / w, 0, w - 1).astype(np.int32)
        y_b = np.clip(y + dy * shift_b / h, 0, h - 1).astype(np.int32)
        flare_ca[:, :, 2] = flare[y_b, x_b, 2]

        return flare_ca

    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.debug("Lens flare module cleaned up")
