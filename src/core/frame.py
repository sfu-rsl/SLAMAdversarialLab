"""Frame data structure for SLAMAdverserialLab."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class Frame:
    """Container for a single dataset frame."""

    # Core data
    image: np.ndarray
    timestamp: float
    sequence_id: str
    frame_idx: int

    # Optional data
    depth: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached properties
    _image_shape: Optional[Tuple[int, int, int]] = field(default=None, init=False, repr=False)
    _depth_shape: Optional[Tuple[int, int]] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Validate and cache frame properties after initialization."""
        self.validate()
        self._cache_shapes()

    def validate(self) -> None:
        """
        Validate frame data.

        Raises:
            ValueError: If frame data is invalid
        """
        if not isinstance(self.image, np.ndarray):
            raise ValueError(f"Image must be numpy array, got {type(self.image)}")

        if self.image.ndim != 3:
            raise ValueError(f"Image must have 3 dimensions (H, W, C), got shape {self.image.shape}")

        if self.image.shape[2] not in [1, 3, 4]:
            raise ValueError(
                f"Image must have 1, 3, or 4 channels, got {self.image.shape[2]} channels"
            )

        if self.image.dtype != np.uint8:
            logger.warning(f"Image dtype is {self.image.dtype}, expected uint8")

        if self.depth is not None:
            if not isinstance(self.depth, np.ndarray):
                raise ValueError(f"Depth must be numpy array, got {type(self.depth)}")

            if self.depth.ndim != 2:
                raise ValueError(f"Depth must have 2 dimensions (H, W), got shape {self.depth.shape}")

            if self.depth.shape[:2] != self.image.shape[:2]:
                raise ValueError(
                    f"Depth shape {self.depth.shape} doesn't match "
                    f"image spatial dimensions {self.image.shape[:2]}"
                )

            if self.depth.dtype != np.float32:
                logger.warning(f"Depth dtype is {self.depth.dtype}, expected float32")

        if not isinstance(self.timestamp, (int, float)):
            raise ValueError(f"Timestamp must be numeric, got {type(self.timestamp)}")

        if not isinstance(self.frame_idx, int):
            raise ValueError(f"Frame index must be integer, got {type(self.frame_idx)}")

        if self.frame_idx < 0:
            raise ValueError(f"Frame index must be non-negative, got {self.frame_idx}")

    def _cache_shapes(self) -> None:
        """Cache shape information for quick access."""
        self._image_shape = self.image.shape
        if self.depth is not None:
            self._depth_shape = self.depth.shape

    @property
    def height(self) -> int:
        """Get frame height."""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """Get frame width."""
        return self.image.shape[1]

    @property
    def channels(self) -> int:
        """Get number of image channels."""
        return self.image.shape[2]

    @property
    def has_depth(self) -> bool:
        """Check if frame has depth data."""
        return self.depth is not None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert frame to dictionary format.

        Returns:
            Dictionary containing frame data
        """
        data = {
            'image': self.image,
            'timestamp': self.timestamp,
            'sequence_id': self.sequence_id,
            'frame_id': self.frame_idx,  # Use 'frame_id' for backward compatibility
            'depth': self.depth
        }

        if self.metadata:
            data.update(self.metadata)

        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Frame':
        """
        Create Frame from dictionary.

        Args:
            data: Dictionary containing frame data

        Returns:
            Frame instance
        """
        # Extract core fields
        image = data.get('image')
        if image is None:
            raise ValueError("Dictionary must contain 'image' field")

        timestamp = data.get('timestamp', 0.0)
        sequence_id = data.get('sequence_id', 'unknown')
        frame_idx = data.get('frame_idx') or data.get('frame_id', 0)

        # Extract optional fields
        depth = data.get('depth')

        # Remaining fields go to metadata
        metadata = {}
        core_fields = {'image', 'timestamp', 'sequence_id', 'frame_idx', 'frame_id', 'depth'}
        for key, value in data.items():
            if key not in core_fields:
                metadata[key] = value

        return cls(
            image=image,
            timestamp=timestamp,
            sequence_id=sequence_id,
            frame_idx=frame_idx,
            depth=depth,
            metadata=metadata
        )

    def __repr__(self) -> str:
        """String representation of the frame."""
        depth_info = f", depth={self.depth.shape}" if self.has_depth else ""
        metadata_info = f", metadata={len(self.metadata)} items" if self.metadata else ""

        return (
            f"Frame("
            f"seq='{self.sequence_id}', "
            f"idx={self.frame_idx}, "
            f"time={self.timestamp:.6f}, "
            f"image={self.image.shape}"
            f"{depth_info}"
            f"{metadata_info})"
        )

    def __eq__(self, other) -> bool:
        """Check equality with another frame."""
        if not isinstance(other, Frame):
            return False

        if (self.timestamp != other.timestamp or
            self.sequence_id != other.sequence_id or
            self.frame_idx != other.frame_idx):
            return False

        if not np.array_equal(self.image, other.image):
            return False

        if self.has_depth != other.has_depth:
            return False
        if self.has_depth and not np.array_equal(self.depth, other.depth):
            return False

        if self.metadata != other.metadata:
            return False

        return True
