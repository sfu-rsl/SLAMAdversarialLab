"""Base class for video encoding-based perturbation modules."""

import os
import subprocess
import shutil
import cv2
import numpy as np
from abc import abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

from ..base import PerturbationModule
from ...utils import get_logger, create_temp_dir


class VideoCodec(str, Enum):
    """Supported video codecs for encoding."""
    LIBX264 = "libx264"
    """H.264 codec (widely compatible)"""
    LIBX265 = "libx265"
    """H.265/HEVC codec (better compression, slower)"""


class EncodingPreset(str, Enum):
    """FFmpeg encoding presets (speed vs compression tradeoff)."""
    ULTRAFAST = "ultrafast"
    """Fastest encoding, largest file size"""
    SUPERFAST = "superfast"
    """Very fast encoding"""
    VERYFAST = "veryfast"
    """Fast encoding"""
    FASTER = "faster"
    """Faster than default"""
    FAST = "fast"
    """Slightly faster than default"""
    MEDIUM = "medium"
    """Default balanced preset"""
    SLOW = "slow"
    """Better compression, slower"""
    SLOWER = "slower"
    """Even better compression"""
    VERYSLOW = "veryslow"
    """Best compression, slowest"""

logger = get_logger(__name__)


class VideoEncodingModuleBase(PerturbationModule):
    """Abstract base class for video encoding-based perturbations.

    Subclasses must implement:
    - _setup_encoding_params(): Parse encoding-specific parameters
    - _get_encoding_flags(): Return ffmpeg encoding flags
    - _get_log_info(): Return info string for logging

    The encoding flow is:
    1. On first _apply() call, encode entire source sequence through video codec
    2. Decode back to frames and store in memory
    3. Return pre-compressed frame for each subsequent _apply() call

    When used in a CompositeModule, the setup context input path can be overridden
    to process images from a preceding module instead of the original dataset.
    """

    requires_full_sequence = True

    def _setup(self, context) -> None:
        """Setup video encoding module."""
        # Common parameters
        params = self.parameters or {}
        self.codec = params.get('codec', 'libx264')
        self.encoding_preset = params.get('encoding_preset', 'medium')
        self.framerate = params.get('framerate', 30)

        # Let subclass parse encoding-specific parameters
        self._setup_encoding_params(params)

        self._setup_complete = self._get_source_path() is not None

        # Verify ffmpeg is available
        if not shutil.which('ffmpeg'):
            raise RuntimeError(
                "ffmpeg not found. Please install ffmpeg: "
                "sudo apt install ffmpeg (Ubuntu) or brew install ffmpeg (macOS)"
            )

        # Will be populated on first apply() - separate cache per camera for stereo support
        self._compressed_frames: Dict[str, List[np.ndarray]] = {}  # camera -> frames
        self._temp_dirs: Dict[str, Path] = {}  # camera -> temp_dir
        self._is_grayscale: bool = False  # Detected on first encoding

        logger.info(f"{self.__class__.__name__} initialized:")
        logger.info(f"  Codec: {self.codec}")
        logger.info(f"  Encoding preset: {self.encoding_preset}")
        logger.info(f"  Framerate: {self.framerate}")
        logger.info(f"  {self._get_log_info()}")

    @abstractmethod
    def _setup_encoding_params(self, params: dict) -> None:
        """Parse encoding-specific parameters from config.

        Args:
            params: Parameters dictionary from config
        """
        pass

    @abstractmethod
    def _get_encoding_flags(self) -> List[str]:
        """Return ffmpeg encoding flags specific to this module.

        Returns:
            List of ffmpeg arguments (e.g., ['-crf', '35'] or ['-b:v', '2M'])
        """
        pass

    @abstractmethod
    def _get_log_info(self) -> str:
        """Return encoding-specific info string for logging.

        Returns:
            String describing the encoding parameters
        """
        pass

    def _reset_encoding_cache(self) -> None:
        """Reset cached encoded frames and temporary artifacts."""
        self._compressed_frames.clear()
        for camera, temp_dir in list(self._temp_dirs.items()):
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        self._temp_dirs.clear()

    def _on_context_updated(self, previous_context, context, reason: str) -> None:
        """Rebind source paths and clear encoding cache on context updates."""
        self._setup_complete = self._get_source_path() is not None
        self._reset_encoding_cache()

    def _apply(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        frame_idx: int,
        camera: str,
        **kwargs
    ) -> np.ndarray:
        """Return pre-compressed frame.

        On first call, encodes entire source sequence through video codec.

        Args:
            image: Input image (used as fallback if frame_idx out of range)
            depth: Depth map (unused)
            frame_idx: Frame index
            camera: Camera role ("left" or "right")
            **kwargs: Additional context (unused)

        Returns:
            Compressed frame from the pre-encoded sequence
        """
        # Ensure setup is complete (source path must be available in context)
        if not self._setup_complete:
            raise RuntimeError(
                f"{self.__class__.__name__} '{self.name}' setup not complete. "
                "Setup context must include dataset_path or input_path. "
                "This should happen automatically - please report this as a bug."
            )

        # Encode frames for this camera if not already cached (supports stereo)
        if camera not in self._compressed_frames:
            self._encode_source_sequence(camera)

        if frame_idx >= len(self._compressed_frames[camera]):
            logger.warning(
                f"Frame {frame_idx} out of range for camera {camera} "
                f"(have {len(self._compressed_frames[camera])} frames), returning original"
            )
            return image

        return self._compressed_frames[camera][frame_idx]

    def _encode_source_sequence(self, camera: str = "left") -> None:
        """Encode entire source sequence through video codec."""
        if self._dataset is None:
            raise RuntimeError(
                "Dataset object not set. setup(context) must include a dataset instance."
            )

        frame_files = self._dataset.get_image_paths(camera)

        if not frame_files:
            dataset_len = len(self._dataset)
            if dataset_len > 0:
                raise RuntimeError(
                    f"Dataset has {dataset_len} frames but get_image_paths('{camera}') returned no files. "
                    f"Dataset metadata is likely malformed."
                )
            raise FileNotFoundError(f"No image files found in dataset for camera {camera}")

        first_img = cv2.imread(str(frame_files[0]), cv2.IMREAD_UNCHANGED)
        self._is_grayscale = len(first_img.shape) == 2 or (len(first_img.shape) == 3 and first_img.shape[2] == 1)
        if self._is_grayscale:
            logger.info(f"Detected grayscale source images")

        logger.info(f"Encoding {len(frame_files)} {camera} frames with {self._get_log_info()}")

        temp_dir = create_temp_dir(prefix=f"video_enc_{camera}_")
        self._temp_dirs[camera] = temp_dir
        temp_video = temp_dir / "encoded.mp4"
        decoded_dir = temp_dir / "decoded"
        decoded_dir.mkdir()

        input_dir = temp_dir / "input"
        input_dir.mkdir()
        for i, frame in enumerate(frame_files):
            link_path = input_dir / f"frame_{i:06d}.png"
            link_path.symlink_to(frame.resolve())

        pix_fmt = 'gray' if self._is_grayscale else 'yuv420p'

        encode_cmd = [
            'ffmpeg', '-y',
            '-framerate', str(self.framerate),
            '-i', str(input_dir / 'frame_%06d.png'),
            '-c:v', self.codec,
        ]
        encode_cmd.extend(self._get_encoding_flags())
        encode_cmd.extend([
            '-preset', self.encoding_preset,
            '-pix_fmt', pix_fmt,
            str(temp_video)
        ])

        # that can cause ffmpeg to fail with libopenh264 errors
        env = {
            'PATH': '/usr/bin:/bin:/usr/local/bin',
            'HOME': os.environ.get('HOME', '/tmp'),
        }

        logger.debug(f"Encode cmd: {' '.join(encode_cmd)}")
        result = subprocess.run(encode_cmd, capture_output=True, env=env)
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg encode failed: {error_msg}")

        # Decode video back to frames
        decode_cmd = [
            'ffmpeg', '-y',
            '-i', str(temp_video),
            str(decoded_dir / "frame_%06d.png")
        ]

        logger.debug(f"Decode cmd: {' '.join(decode_cmd)}")
        result = subprocess.run(decode_cmd, capture_output=True, env=env)  # Use same clean env
        if result.returncode != 0:
            error_msg = result.stderr.decode() if result.stderr else "Unknown error"
            raise RuntimeError(f"ffmpeg decode failed: {error_msg}")

        decoded_files = sorted(decoded_dir.glob("frame_*.png"))
        frames_list = []
        for f in decoded_files:
            if self._is_grayscale:
                # Read as grayscale to preserve original format
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            else:
                img = cv2.imread(str(f))
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if img is None:
                logger.warning(f"Failed to read decoded frame: {f}")
                continue
            frames_list.append(img)

        # Store frames in camera-specific cache
        self._compressed_frames[camera] = frames_list

        logger.info(
            f"Loaded {len(frames_list)} compressed {camera} frames into memory"
        )

        # Clean up temp video file (keep decoded frames until cleanup)
        temp_video.unlink(missing_ok=True)

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._compressed_frames.clear()
        for camera, temp_dir in list(self._temp_dirs.items()):
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir)
        self._temp_dirs.clear()
        logger.debug(f"{self.__class__.__name__} cleaned up")
