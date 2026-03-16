"""Frame dropping / temporal discontinuity perturbation."""

import numpy as np
from typing import Optional, List, Set
from dataclasses import dataclass, field

from ..base import PerturbationModule
from ...robustness.param_spec import BoundaryParamSpec
from ...utils import get_logger

logger = get_logger(__name__)


# Presets for common drop patterns
FRAME_DROP_PRESETS = {
    "light": {"mode": "random", "drop_rate": 0.05},
    "medium": {"mode": "random", "drop_rate": 0.15},
    "heavy": {"mode": "random", "drop_rate": 0.30},
    "periodic_10": {"mode": "periodic", "drop_interval": 10},
    "periodic_5": {"mode": "periodic", "drop_interval": 5},
}


@dataclass
class FrameDropParameters:
    """Parameters for frame dropping simulation."""

    mode: str = field(default="random", metadata={"choices": ["random", "periodic"]})
    """Drop mode: 'random' or 'periodic'"""

    drop_rate: float = 0.1
    """For random mode: probability of dropping each frame (0.0-1.0)"""

    drop_interval: int = 10
    """For periodic mode: drop every Nth frame"""

    seed: Optional[int] = None
    """Random seed for reproducibility"""


class FrameDropModule(PerturbationModule):
    """Frame dropping / temporal discontinuity perturbation."""

    module_name = "frame_drop"
    module_description = "Frame dropping simulation for temporal discontinuities"

    PARAMETERS_CLASS = FrameDropParameters
    SEARCHABLE_PARAMS = {
        "drop_rate": BoundaryParamSpec(domain="integer", canonicalize=lambda value: float(value) / 100.0),
    }

    def __init__(self, config):
        """Initialize frame dropping module.

        Args:
            config: PerturbationConfig with frame drop parameters
        """
        super().__init__(config)

        # Parse parameters from config
        parameters = self.parameters if self.parameters else {}

        preset = parameters.get("preset")
        if preset:
            if preset not in FRAME_DROP_PRESETS:
                raise ValueError(
                    f"Unknown preset '{preset}'. Available: {list(FRAME_DROP_PRESETS.keys())}"
                )
            preset_params = FRAME_DROP_PRESETS[preset].copy()
            # Allow overrides
            for key, value in parameters.items():
                if key != "preset":
                    preset_params[key] = value
            parameters = preset_params

        self.params = FrameDropParameters(
            mode=parameters.get("mode", "random"),
            drop_rate=parameters.get("drop_rate", 0.1),
            drop_interval=parameters.get("drop_interval", 10),
            seed=parameters.get("seed"),
        )

        if self.params.mode not in ("random", "periodic"):
            raise ValueError(f"Invalid mode '{self.params.mode}'. Must be 'random' or 'periodic'")
        if not 0.0 <= self.params.drop_rate <= 1.0:
            raise ValueError(f"drop_rate must be between 0 and 1, got {self.params.drop_rate}")
        if self.params.drop_interval < 1:
            raise ValueError(f"drop_interval must be >= 1, got {self.params.drop_interval}")

        # Random state for reproducibility
        self._rng: Optional[np.random.Generator] = None

        # Track which frames are dropped (for logging/output)
        self._dropped_frames: List[int] = []

        # For random mode with seed, pre-compute decisions for consistency
        self._drop_decisions: Optional[Set[int]] = None
        self._primary_camera: Optional[str] = None

    def _setup(self, context) -> None:
        """Initialize frame dropping module."""
        logger.info(f"  Mode: {self.params.mode}")

        if self.params.mode == "random":
            logger.info(f"  Drop rate: {self.params.drop_rate * 100:.1f}%")
        else:
            logger.info(f"  Drop interval: every {self.params.drop_interval} frames")

        if self.params.seed is not None:
            logger.info(f"  Seed: {self.params.seed}")
            self._rng = np.random.default_rng(self.params.seed)
        else:
            self._rng = np.random.default_rng()

        # Reset tracking
        self._dropped_frames = []
        self._drop_decisions = None
        self._primary_camera = None

    def _should_drop(self, frame_idx: int) -> bool:
        """Determine if a frame should be dropped.

        Args:
            frame_idx: Frame index

        Returns:
            True if frame should be dropped
        """
        if self.params.mode == "periodic":
            # Drop every Nth frame (0-indexed, so drop when idx % interval == interval-1)
            return (frame_idx + 1) % self.params.drop_interval == 0
        else:
            # Random mode
            if self.params.seed is not None:
                # With seed: use deterministic per-frame decision
                frame_rng = np.random.default_rng(self.params.seed + frame_idx)
                return frame_rng.random() < self.params.drop_rate
            else:
                # Without seed: use the module's RNG
                return self._rng.random() < self.params.drop_rate

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str = "left",
        **kwargs
    ) -> Optional[np.ndarray]:
        """Apply frame dropping.

        Args:
            image: Input image (H, W, 3) in range [0, 255], uint8
            depth: Depth map (unused)
            frame_idx: Frame index
            camera: Camera identifier (for stereo sync logging)
            **kwargs: Additional context (unused)

        Returns:
            Original image if kept, None if dropped
        """
        # Decide on the first camera we observe and reuse that decision for any others
        # so stereo pairs stay synchronized regardless of camera naming convention.
        if self._primary_camera is None:
            self._primary_camera = camera

        if camera == self._primary_camera:
            should_drop = self._should_drop(frame_idx)
            if should_drop:
                self._dropped_frames.append(frame_idx)
                logger.debug(f"Frame {frame_idx} dropped")
        else:
            # For non-primary cameras, reuse the primary-camera decision for this frame.
            should_drop = frame_idx in self._dropped_frames

        if should_drop:
            return None

        return image

    def get_dropped_frames(self) -> List[int]:
        """Get list of dropped frame indices.

        Returns:
            List of frame indices that were dropped
        """
        return self._dropped_frames.copy()

    def _cleanup(self) -> None:
        """Clean up frame dropping module resources."""
        if self._dropped_frames:
            logger.info(f"Total frames dropped: {len(self._dropped_frames)}")
            logger.debug(f"Dropped frame indices: {self._dropped_frames}")
