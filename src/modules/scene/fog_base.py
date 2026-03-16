"""Shared base classes for fog perturbation modules."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class FogParametersBase:
    """Base parameters for fog simulation shared by all fog modules."""

    visibility_m: float = 50.0
    """Visibility distance in meters"""

    atmospheric_light: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    """RGB values for atmospheric light (fog color), range [0, 1]"""

    min_depth_m: float = 0.1
    """Minimum depth value in meters"""

    max_depth_m: float = 100.0
    """Maximum depth value in meters"""

    incremental: bool = False
    """Enable progressive fog (clear to dense)"""

    start_visibility_m: Optional[float] = None
    """Starting visibility for incremental fog (meters)"""

    end_visibility_m: Optional[float] = None
    """Ending visibility for incremental fog (meters)"""

    total_frames: Optional[int] = None
    """Total number of frames for incremental interpolation"""

    @property
    def beta(self) -> float:
        """Calculate scattering coefficient from visibility.

        Uses Koschmieder's law: V = 3.912 / beta
        where V is visibility in meters.
        """
        return 3.912 / self.visibility_m


class FogPresetsBase:
    """Base class for predefined fog intensity presets."""

    # Preset definitions (subclasses override with their specific parameter classes)
    LIGHT_VISIBILITY = 200.0
    LIGHT_ATMOSPHERIC_LIGHT = (0.85, 0.85, 0.85)

    MEDIUM_VISIBILITY = 50.0
    MEDIUM_ATMOSPHERIC_LIGHT = (0.8, 0.8, 0.8)

    HEAVY_VISIBILITY = 20.0
    HEAVY_ATMOSPHERIC_LIGHT = (0.75, 0.75, 0.75)

    DENSE_VISIBILITY = 10.0
    DENSE_ATMOSPHERIC_LIGHT = (0.7, 0.7, 0.7)


class FogModuleBase(PerturbationModule):
    """Base class for fog perturbation modules using Koschmieder model.

    Implements the Koschmieder fog model:
    - Transmission: t(x) = exp(-beta * depth(x))
    - Foggy image: I(x) = J(x) * t(x) + A * (1 - t(x))

    Where:
    - J(x) is the clear scene radiance (original image)
    - t(x) is the transmission map
    - A is the atmospheric light
    - beta is the scattering coefficient

    Subclasses must implement:
    - _setup(): Module-specific initialization
    - _apply(): Apply fog effect using depth from various sources
    - params: FogParametersBase or subclass instance
    """

    # Subclasses must set this in _setup()
    params: FogParametersBase

    def _interpolate_visibility(self, frame_idx: int) -> float:
        """Calculate visibility for incremental fog based on frame index.

        Args:
            frame_idx: Current frame index

        Returns:
            Interpolated visibility in meters
        """
        if not self.params.incremental:
            return self.params.visibility_m

        start = self.params.start_visibility_m
        end = self.params.end_visibility_m

        if self.params.total_frames is not None:
            # Bounded interpolation with known total frames
            frame_idx = min(frame_idx, self.params.total_frames - 1)
            progress = frame_idx / max(1, self.params.total_frames - 1)
        else:
            # Unbounded interpolation (experimental)
            progress = min(1.0, frame_idx / 1000.0)

        # Linear interpolation
        visibility = start + (end - start) * progress
        return visibility

    def _get_current_fog_params(self, frame_idx: int) -> Tuple[float, float]:
        """Get current visibility and beta for a frame.

        Handles incremental fog interpolation.

        Args:
            frame_idx: Current frame index

        Returns:
            Tuple of (visibility_m, beta)
        """
        if self.params.incremental:
            # Auto-detect total_frames on first call if not provided
            if self.params.total_frames is None:
                if not hasattr(self, '_warned_no_total_frames'):
                    logger.warning(
                        f"Incremental fog enabled but total_frames not provided. "
                        f"Will use frame_idx as progress indicator (unbounded interpolation)."
                    )
                    self._warned_no_total_frames = True

            current_visibility = self._interpolate_visibility(frame_idx)
            current_beta = 3.912 / current_visibility

            logger.debug(
                f"Frame {frame_idx}: Incremental fog visibility={current_visibility:.1f}m, "
                f"beta={current_beta:.4f}"
            )
        else:
            current_visibility = self.params.visibility_m
            current_beta = self.params.beta

        return current_visibility, current_beta

    def _apply_koschmieder_model(
        self,
        image_float: np.ndarray,
        transmission: np.ndarray
    ) -> np.ndarray:
        """Apply the Koschmieder fog model to an image.

        Args:
            image_float: Input image as float32 in range [0, 1], shape (H, W, 3)
            transmission: Transmission map in range [0, 1], shape (H, W) or (H, W, 1)

        Returns:
            Foggy image as uint8 in range [0, 255], shape (H, W, 3)
        """
        # Expand transmission to match image channels if needed
        if transmission.ndim == 2:
            transmission = np.expand_dims(transmission, axis=-1)

        atmospheric_light = np.array(
            self.params.atmospheric_light,
            dtype=np.float32
        ).reshape(1, 1, 3)

        foggy_image = (
            image_float * transmission +
            atmospheric_light * (1 - transmission)
        )

        foggy_image = np.clip(foggy_image * 255, 0, 255).astype(np.uint8)

        return foggy_image

    def _validate_incremental_config(self) -> None:
        """Validate incremental fog configuration.

        Raises:
            ValueError: If incremental config is invalid
        """
        if self.params.incremental:
            if self.params.start_visibility_m is None or self.params.end_visibility_m is None:
                raise ValueError(
                    "Incremental fog requires both 'start_visibility_m' and 'end_visibility_m' parameters"
                )
            if self.params.start_visibility_m <= 0 or self.params.end_visibility_m <= 0:
                raise ValueError(
                    f"Visibility values must be positive (got start={self.params.start_visibility_m}, "
                    f"end={self.params.end_visibility_m})"
                )
            logger.info(
                f"Incremental fog enabled: {self.params.start_visibility_m:.1f}m -> "
                f"{self.params.end_visibility_m:.1f}m"
            )

    def get_config(self) -> Dict[str, Any]:
        """Get module configuration.

        Returns:
            Configuration dictionary with common fog parameters
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__.lower().replace('module', ''),
            'enabled': self.enabled,
            'visibility_m': self.params.visibility_m,
            'beta': self.params.beta,
            'atmospheric_light': self.params.atmospheric_light,
            'min_depth_m': self.params.min_depth_m,
            'max_depth_m': self.params.max_depth_m
        }
