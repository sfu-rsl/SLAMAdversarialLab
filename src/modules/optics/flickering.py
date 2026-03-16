"""Brightness and contrast flicker perturbation."""

import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass

from ..base import PerturbationModule
from ...utils import get_logger

logger = get_logger(__name__)


@dataclass
class FlickerParameters:
    """Parameters for flickering simulation."""

    intensity: float = 0.3
    """Flicker intensity (0-1), where 0=no flicker, 1=maximum flicker"""

    frequency: float = 0.1
    """Flicker frequency in Hz (cycles per frame)"""

    brightness_range: tuple = (-0.3, 0.3)
    """Range of brightness adjustment (min, max) relative to original"""

    contrast_range: tuple = (0.7, 1.3)
    """Range of contrast adjustment (min, max) multiplier"""

    affect_brightness: bool = True
    """Apply brightness flickering"""

    affect_contrast: bool = True
    """Apply contrast flickering"""

    strobe_duty_cycle: float = 0.5
    """Duty cycle for strobe pattern (0-1), fraction of time 'on'"""


class FlickerModule(PerturbationModule):
    """Brightness and contrast flicker perturbation."""

    module_name = "flickering"
    module_description = "Brightness/contrast flickering simulation"

    PARAMETERS_CLASS = FlickerParameters

    def __init__(self, config):
        """Initialize flickering module.

        Args:
            config: PerturbationConfig with flickering parameters
        """
        from ...config.schema import PerturbationConfig
        super().__init__(config)

        # Parse parameters from config
        parameters = self.parameters if self.parameters else {}

        self.params = FlickerParameters(
            intensity=parameters.get('intensity', 0.3),
            frequency=parameters.get('frequency', 0.1),
            brightness_range=tuple(parameters.get('brightness_range', (-0.3, 0.3))),
            contrast_range=tuple(parameters.get('contrast_range', (0.7, 1.3))),
            affect_brightness=parameters.get('affect_brightness', True),
            affect_contrast=parameters.get('affect_contrast', True),
            strobe_duty_cycle=parameters.get('strobe_duty_cycle', 0.5)
        )

    def _setup(self, context) -> None:
        """Initialize flickering module."""
        logger.info(f"  Intensity: {self.params.intensity}")
        logger.info(f"  Frequency: {self.params.frequency} Hz")
        logger.info(f"  Strobe duty cycle: {self.params.strobe_duty_cycle}")
        logger.info(f"  Affect brightness: {self.params.affect_brightness}")
        logger.info(f"  Affect contrast: {self.params.affect_contrast}")

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str = "left",
        **kwargs
    ) -> np.ndarray:
        """Apply flickering effect to image.

        Args:
            image: Input image (H, W, 3) in range [0, 255], uint8
            depth: Depth map (unused for flickering)
            frame_idx: Frame index for temporal patterns
            **kwargs: Additional context (unused)

        Returns:
            Flickered image (H, W, 3) in range [0, 255], uint8
        """

        img_float = image.astype(np.float32)

        # Calculate flicker factors based on pattern
        brightness_factor, contrast_factor = self._calculate_flicker_factors(frame_idx)

        if self.params.affect_brightness:
            img_float = img_float + brightness_factor * 255.0

        if self.params.affect_contrast:
            # Contrast adjustment around midpoint (128)
            img_float = (img_float - 128.0) * contrast_factor + 128.0

        # Clamp to valid range and convert back to uint8
        img_float = np.clip(img_float, 0, 255)
        return img_float.astype(np.uint8)

    def _calculate_flicker_factors(self, frame_idx: int) -> tuple:
        """Generate strobe (on/off) flicker factors.

        Args:
            frame_idx: Current frame index

        Returns:
            Tuple of (brightness_factor, contrast_factor)
        """
        # Calculate position in cycle
        cycle_length = 1.0 / self.params.frequency
        position_in_cycle = (frame_idx % cycle_length) / cycle_length

        # On or off based on duty cycle
        is_on = position_in_cycle < self.params.strobe_duty_cycle

        if is_on:
            # Bright phase
            brightness = self.params.brightness_range[1] * self.params.intensity
            contrast = self.params.contrast_range[1]
            contrast = 1.0 + (contrast - 1.0) * self.params.intensity
        else:
            # Dark phase
            brightness = self.params.brightness_range[0] * self.params.intensity
            contrast = self.params.contrast_range[0]
            contrast = 1.0 + (contrast - 1.0) * self.params.intensity

        return brightness, contrast

    def _cleanup(self) -> None:
        """Clean up flickering module resources."""
        logger.debug(f"Cleaning up {self.name} module")
