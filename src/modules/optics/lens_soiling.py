"""Lens soiling perturbation with static dark spots."""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class LensSoilingParameters:
    """Parameters for lens soiling simulation."""

    num_particles: int = 30
    """Number of dust particles"""

    size_range: Tuple[int, int] = (10, 40)
    """Bokeh size range in pixels (min, max)"""

    opacity_range: Tuple[float, float] = (0.3, 0.6)
    """Opacity range for dust particles (0-1), how much light is blocked"""

    color: Tuple[int, int, int] = (40, 35, 30)
    """Dust color RGB (0-255), dark brownish-gray by default"""

    seed: Optional[int] = None
    """Random seed for reproducibility"""


class LensSoilingPresets:
    """Common lens soiling presets."""

    @staticmethod
    def get_preset(name: str) -> LensSoilingParameters:
        """Get preset configuration by name.

        Args:
            name: Preset name

        Returns:
            LensSoilingParameters for the preset

        Raises:
            ValueError: If preset name is unknown
        """
        presets = {
            # Light soiling - subtle effect
            "light_soiling": LensSoilingParameters(
                num_particles=15,
                size_range=(8, 25),
                opacity_range=(0.2, 0.4),
                color=(50, 45, 40)  # Light gray-brown
            ),

            # Medium soiling - noticeable but not severe
            "medium_soiling": LensSoilingParameters(
                num_particles=40,
                size_range=(12, 40),
                opacity_range=(0.3, 0.55),
                color=(40, 35, 30)  # Medium gray-brown
            ),

            # Heavy soiling - significant visual degradation
            "heavy_soiling": LensSoilingParameters(
                num_particles=80,
                size_range=(15, 60),
                opacity_range=(0.4, 0.7),
                color=(30, 25, 20)  # Dark brown
            ),
        }

        if name not in presets:
            raise ValueError(
                f"Unknown lens soiling preset: {name}. "
                f"Available presets: {', '.join(presets.keys())}"
            )

        return presets[name]


class LensSoilingModule(PerturbationModule):
    """Lens soiling perturbation with static dark spots."""

    module_name = "lens_soiling"
    module_description = "Lens soiling with bokeh effect"

    PARAMETERS_CLASS = LensSoilingParameters

    def _setup(self, context) -> None:
        """Setup lens soiling module."""
        params = self.parameters or {}

        if 'preset' in params:
            preset_name = params['preset']
            preset_params = LensSoilingPresets.get_preset(preset_name)

            self.num_particles = preset_params.num_particles
            self.size_range = preset_params.size_range
            self.opacity_range = preset_params.opacity_range
            self.color = preset_params.color
            self.seed = preset_params.seed

            # Allow preset override with explicit parameters
            self.num_particles = params.get('num_particles', self.num_particles)
            self.size_range = tuple(params.get('size_range', self.size_range))
            self.opacity_range = tuple(params.get('opacity_range', self.opacity_range))
            self.color = tuple(params.get('color', self.color))
            self.seed = params.get('seed', self.seed)

        else:
            # No preset - use explicit parameters or defaults
            self.num_particles = params.get('num_particles', 30)
            self.size_range = tuple(params.get('size_range', [10, 40]))
            self.opacity_range = tuple(params.get('opacity_range', [0.3, 0.6]))
            self.color = tuple(params.get('color', [40, 35, 30]))
            self.seed = params.get('seed', None)

        self.rng = np.random.RandomState(self.seed)

        self._soiling_mask = None
        self._cached_shape = None

        logger.info(f"Lens soiling module initialized:")
        logger.info(f"  Particles: {self.num_particles}")
        logger.info(f"  Size range: {self.size_range} px")
        logger.info(f"  Opacity range: {self.opacity_range}")
        logger.info(f"  Color: RGB{self.color}")
        logger.info(f"  Seed: {self.seed or 'random'}")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray] = None,
        frame_idx: int = 0,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply soiling effect to image.

        Args:
            image: Input image (H, W, 3) in RGB format, uint8
            depth: Depth map (unused for lens soiling)
            frame_idx: Frame index (unused - soiling is static)
            camera: Camera identifier
            **kwargs: Additional context (unused)

        Returns:
            Image with soiling effect applied
        """
        h, w = image.shape[:2]

        # Generate soiling mask if not cached or shape changed
        if self._soiling_mask is None or self._cached_shape != (h, w):
            self._generate_soiling_mask(h, w)
            self._cached_shape = (h, w)

        # Where alpha is the soiling mask (opacity of soiling particles)
        soiling_color = np.array(self.color, dtype=np.float32).reshape(1, 1, 3)

        img_float = image.astype(np.float32)
        alpha = self._soiling_mask[:, :, np.newaxis]  # (H, W, 1)

        result = img_float * (1 - alpha) + soiling_color * alpha
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _generate_soiling_mask(self, h: int, w: int) -> None:
        """Generate soiling particle mask with bokeh effect.

        Args:
            h: Image height
            w: Image width
        """
        mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(self.num_particles):
            cx = self.rng.randint(0, w)
            cy = self.rng.randint(0, h)

            radius = self.rng.randint(self.size_range[0], self.size_range[1] + 1)

            opacity = self.rng.uniform(self.opacity_range[0], self.opacity_range[1])

            self._add_soiling_particle(mask, cx, cy, radius, opacity)

        self._soiling_mask = mask
        logger.debug(f"Generated soiling mask with {self.num_particles} particles")

    def _add_soiling_particle(
        self,
        mask: np.ndarray,
        cx: int,
        cy: int,
        radius: int,
        opacity: float
    ) -> None:
        """Add a single dust particle to the mask.

        Args:
            mask: Mask array to modify in-place
            cx: Center X coordinate
            cy: Center Y coordinate
            radius: Bokeh radius in pixels
            opacity: Particle opacity (0-1)
        """
        h, w = mask.shape

        x_min = max(0, cx - radius * 2)
        x_max = min(w, cx + radius * 2)
        y_min = max(0, cy - radius * 2)
        y_max = min(h, cy + radius * 2)

        if x_min >= x_max or y_min >= y_max:
            return

        y_local, x_local = np.ogrid[y_min:y_max, x_min:x_max]

        dist = np.sqrt((x_local - cx)**2 + (y_local - cy)**2)

        sigma = radius / 2.0
        bokeh = np.exp(-(dist**2) / (2 * sigma**2))
        bokeh = bokeh * opacity

        mask[y_min:y_max, x_min:x_max] = np.maximum(
            mask[y_min:y_max, x_min:x_max],
            bokeh
        )

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._soiling_mask = None
        self._cached_shape = None
        logger.debug("Lens soiling module cleaned up")
