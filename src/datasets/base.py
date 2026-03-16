"""Base class for dataset loaders."""

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union, Iterator
from pathlib import Path
import numpy as np

from ..utils import get_logger, copy_or_truncate_text_file
from ..utils.paths import create_temp_dir
from ..config.schema import DatasetConfig
from ..core.frame import Frame

logger = get_logger(__name__)


@dataclass(frozen=True)
class CameraIntrinsics:
    """Typed camera-intrinsics contract returned by dataset adapters."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: Optional[float] = None
    height: Optional[float] = None
    baseline: Optional[float] = None


class Dataset(ABC):
    """
    Abstract base class for all dataset loaders.

    Datasets provide a unified interface for loading images, depth maps,
    and ground truth trajectories from various sources.
    """

    REQUIRED_FRAME_FIELDS: Dict[str, tuple] = {
        "image_path": (str,),
        "timestamp": (int, float),
        "sequence_id": (str,),
        "frame_id": (int,),
    }
    OPTIONAL_FRAME_FIELDS: Dict[str, tuple] = {
        "image_path_right": (str,),
        "depth_path": (str,),
        "rgb_filename": (str,),
        "rgb_filename_right": (str,),
    }

    def __init__(self, config: DatasetConfig):
        """
        Initialize the dataset.

        Args:
            config: Dataset configuration containing path, sequence, etc.
        """
        self.config = config
        self.path = Path(config.path)
        self.max_frames = config.max_frames

        # To be populated by subclasses
        self._frames = []  # List of frame information
        self._ground_truth = None  # Ground truth trajectory

        self._validate_path()
        self._sequence_name = self._resolve_sequence_name()
        self._validate_capabilities_contract(pre_load=True)
        self._load_dataset()
        self._validate_capabilities_contract(pre_load=False)
        self._validate_frame_schema()

        if self.max_frames and len(self._frames) > self.max_frames:
            logger.info(f"Limiting dataset to {self.max_frames} frames")
            self._frames = self._frames[:self.max_frames]

    @property
    def dataset_type(self) -> str:
        """Get the dataset type identifier.

        Returns:
            Dataset type string (e.g., 'kitti', 'tum', 'euroc')
        """
        return self.config.type

    @property
    def sequence_name(self) -> str:
        """Get the canonical sequence name for this dataset instance.

        Returns:
            Configured sequence name when provided; otherwise path-derived sequence name.
        """
        return self._sequence_name

    def _sequence_name_from_path(self) -> str:
        """Infer the sequence label from the resolved dataset path."""
        return self.path.name

    def _normalize_sequence_for_compare(self, sequence: str) -> str:
        """Normalize a sequence label for equality checks."""
        return str(sequence).strip()

    def _resolve_sequence_name(self) -> str:
        """Resolve canonical sequence name and validate config/path consistency."""
        path_sequence = self._sequence_name_from_path()
        if not str(path_sequence).strip():
            raise ValueError(
                f"{self.__class__.__name__} could not infer sequence from dataset path '{self.path}'."
            )

        configured_sequence = (self.config.sequence or "").strip()
        if not configured_sequence:
            return path_sequence

        normalized_config = self._normalize_sequence_for_compare(configured_sequence)
        normalized_path = self._normalize_sequence_for_compare(path_sequence)
        if normalized_config != normalized_path:
            raise ValueError(
                f"{self.__class__.__name__} sequence mismatch: dataset.sequence="
                f"'{configured_sequence}' does not match dataset path '{self.path}' "
                f"(path-derived sequence '{path_sequence}')."
            )
        return configured_sequence

    @property
    def is_stereo(self) -> bool:
        """Check if this dataset is configured for stereo mode.

        Returns:
            True if stereo mode is enabled, False otherwise.
            Subclasses should override this if they support stereo.
        """
        return False  # Default: mono

    @property
    def supports_stereo(self) -> bool:
        """Whether this dataset implementation supports stereo mode at all.

        Returns:
            True when the dataset format has a valid right camera stream.
        """
        return False

    def _get_stereo_contract_state(self) -> Tuple[bool, bool, bool]:
        """Return (requested_stereo, active_stereo, supports_stereo)."""
        requested_stereo = bool(getattr(self.config, "load_stereo", False))
        active_stereo = bool(self.is_stereo)
        supports_stereo = bool(self.supports_stereo)
        return requested_stereo, active_stereo, supports_stereo

    def _resolve_active_camera_roles(self, *, pre_load: bool) -> List[str]:
        """Resolve active camera roles under the stereo contract.

        Args:
            pre_load: True when called before dataset loading is complete.

        Returns:
            Active camera roles for runtime use.
        """
        requested_stereo, active_stereo, supports_stereo = self._get_stereo_contract_state()

        if requested_stereo and not supports_stereo:
            raise ValueError(
                f"{self.__class__.__name__} does not support stereo mode, "
                "but dataset.load_stereo=true was requested."
            )

        if active_stereo and not supports_stereo:
            raise ValueError(
                f"{self.__class__.__name__} reports is_stereo=True but supports_stereo=False."
            )

        if not pre_load and requested_stereo and not active_stereo:
            raise ValueError(
                f"{self.__class__.__name__} received dataset.load_stereo=true but did not "
                "enter stereo mode (is_stereo=False)."
            )

        camera_roles = ["left"]
        if active_stereo:
            camera_roles.append("right")
        return camera_roles

    def get_active_camera_roles(self) -> List[str]:
        """Get active camera roles for this dataset instance.

        Returns:
            A list containing "left" for mono mode, or ["left", "right"] for stereo mode.

        Raises:
            ValueError: If stereo capability contract is inconsistent.
        """
        return list(self._resolve_active_camera_roles(pre_load=False))

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        """
        Resolve the dataset path from configuration.

        This method is called before dataset initialization to resolve
        the path when using sequence-based configuration. Subclasses
        should override this to implement dataset-specific path resolution.

        Args:
            config: Dataset configuration (may have path=None and sequence set)

        Returns:
            Resolved path string

        Raises:
            ValueError: If path cannot be resolved
        """
        if config.path:
            return config.path

        raise ValueError(
            f"Dataset path is required. Either provide 'path' directly "
            f"or use 'sequence' with a dataset type that supports auto-resolution."
        )

    def _validate_path(self) -> None:
        """
        Validate that the dataset path exists.

        Raises:
            FileNotFoundError: If dataset path doesn't exist
        """
        if self.config.type != "mock" and not self.path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.path}")

    @abstractmethod
    def _load_dataset(self) -> None:
        """
        Load the dataset and populate internal structures.

        Subclasses must implement this to:
        - Load frame metadata into self._frames
        - Load ground truth trajectory into self._ground_truth (if available)
        """
        pass

    def __len__(self) -> int:
        """
        Get the total number of frames in the dataset.

        Returns:
            Number of frames
        """
        return len(self._frames)

    def __getitem__(self, idx: int, return_frame: bool = False) -> Union[Dict[str, Any], Frame]:
        """
        Get a single frame from the dataset.

        Args:
            idx: Frame index
            return_frame: If True, return Frame object instead of dict

        Returns:
            Dictionary containing:
                - 'image': RGB image as numpy array (H, W, 3) uint8
                - 'depth': Optional depth map as numpy array (H, W) float32 in meters
                - 'timestamp': Frame timestamp (float)
                - 'sequence_id': Sequence identifier (str)
                - 'frame_id': Frame identifier within sequence (int)
            OR Frame object if return_frame=True

        Raises:
            IndexError: If idx is out of range
        """
        if idx < 0:
            idx = len(self) + idx

        if idx < 0 or idx >= len(self):
            raise IndexError(f"Frame index {idx} out of range [0, {len(self)})")

        frame_data = self._load_frame(idx)

        if return_frame:
            return Frame.from_dict(frame_data)
        else:
            return frame_data

    @abstractmethod
    def _load_frame(self, idx: int) -> Dict[str, Any]:
        """
        Load a specific frame's data.

        Args:
            idx: Valid frame index

        Returns:
            Frame data dictionary
        """
        pass

    def get_ground_truth(self) -> Optional[np.ndarray]:
        """
        Get ground truth trajectory if available.

        Returns:
            Ground truth poses as numpy array of shape (N, 4, 4) representing
            4x4 transformation matrices, or None if not available
        """
        return self._ground_truth

    def get_sequence_info(self) -> Dict[str, Any]:
        """
        Get information about loaded sequences.

        Returns:
            Dictionary containing sequence metadata
        """
        unique_sequences = set(f.get('sequence_id', 'unknown') for f in self._frames)
        return {
            'sequences': list(unique_sequences),
            'total_frames': len(self),
            'frames_per_sequence': self._count_frames_per_sequence()
        }

    def _count_frames_per_sequence(self) -> Dict[str, int]:
        """Count frames in each sequence."""
        counts = {}
        for frame in self._frames:
            seq_id = frame.get('sequence_id', 'unknown')
            counts[seq_id] = counts.get(seq_id, 0) + 1
        return counts

    def __iter__(self):
        """Iterate over all frames in the dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def get_frame(self, idx: int) -> Frame:
        """
        Get a frame as a Frame object.

        Args:
            idx: Frame index

        Returns:
            Frame object

        Raises:
            IndexError: If idx is out of range
        """
        return self.__getitem__(idx, return_frame=True)

    def iter_frames(self):
        """Iterate over frames as Frame objects."""
        for idx in range(len(self)):
            yield self.get_frame(idx)

    def _validate_frame_schema(self) -> None:
        """Validate that loaded frame metadata follows the canonical schema."""
        for idx, frame_info in enumerate(self._frames):
            if not isinstance(frame_info, dict):
                raise ValueError(
                    f"{self.__class__.__name__} frame {idx} has invalid type "
                    f"{type(frame_info).__name__}; expected dict"
                )

            # Required fields
            for field, allowed_types in self.REQUIRED_FRAME_FIELDS.items():
                if field not in frame_info:
                    raise ValueError(
                        f"{self.__class__.__name__} frame {idx} missing required field '{field}'. "
                        f"Available fields: {sorted(frame_info.keys())}"
                    )
                value = frame_info[field]
                if not isinstance(value, allowed_types) or isinstance(value, bool):
                    raise ValueError(
                        f"{self.__class__.__name__} frame {idx} field '{field}' has invalid type "
                        f"{type(value).__name__}; expected {allowed_types}"
                    )

            # Optional fields (if present)
            for field, allowed_types in self.OPTIONAL_FRAME_FIELDS.items():
                if field in frame_info:
                    value = frame_info[field]
                    if not isinstance(value, allowed_types):
                        raise ValueError(
                            f"{self.__class__.__name__} frame {idx} field '{field}' has invalid type "
                            f"{type(value).__name__}; expected {allowed_types}"
                        )

            # Path-like fields must be non-empty strings
            for path_field in ("image_path", "image_path_right", "depth_path"):
                if path_field in frame_info and not str(frame_info[path_field]).strip():
                    raise ValueError(
                        f"{self.__class__.__name__} frame {idx} field '{path_field}' cannot be empty"
                    )

        if self._frames and "right" in self.get_active_camera_roles():
            missing_right = [
                idx for idx, frame in enumerate(self._frames)
                if "image_path_right" not in frame
            ]
            if missing_right:
                first_missing = missing_right[0]
                raise ValueError(
                    f"{self.__class__.__name__} is stereo but frame {first_missing} "
                    f"is missing 'image_path_right' ({len(missing_right)} frame(s) missing)"
                )

    def _validate_capabilities_contract(self, pre_load: bool) -> None:
        """Validate core dataset capability contract.

        This enforces a predictable extension surface for dataset implementers:
        - stereo capability declaration vs requested mode
        - camera directory naming contract
        - destination-aware metadata uniqueness
        - explicit contract usage by runtime modules (no path-heuristic inference)

        Args:
            pre_load: True when called before _load_dataset(), False after.
        """
        requested_stereo, _, _ = self._get_stereo_contract_state()
        camera_roles = self._resolve_active_camera_roles(pre_load=pre_load)

        left_camera = self._validate_camera_directory_name(
            "left",
            self.get_image_directory_name("left"),
        )

        if requested_stereo or "right" in camera_roles:
            try:
                right_camera = self._validate_camera_directory_name(
                    "right",
                    self.get_image_directory_name("right"),
                )
            except Exception as e:
                raise ValueError(
                    f"{self.__class__.__name__} stereo mode requires "
                    "get_image_directory_name('right') to be available."
                ) from e

            if right_camera == left_camera:
                raise ValueError(
                    f"{self.__class__.__name__} maps left and right cameras to the same "
                    f"directory name '{left_camera}'."
                )

        metadata_files = self.get_metadata_files_with_dest()
        destination_to_sources: Dict[str, List[Path]] = {}
        for idx, descriptor in enumerate(metadata_files):
            if not isinstance(descriptor, tuple) or len(descriptor) != 3:
                raise ValueError(
                    f"{self.__class__.__name__}.get_metadata_files_with_dest() entry {idx} "
                    "must be a 3-tuple: (source_path, dest_name, should_truncate)"
                )

            source_path, dest_name, should_truncate = descriptor
            if not isinstance(source_path, Path):
                raise ValueError(
                    f"{self.__class__.__name__}.get_metadata_files_with_dest() entry {idx} "
                    f"source_path must be pathlib.Path, got {type(source_path).__name__}"
                )
            if not isinstance(dest_name, str) or not dest_name.strip():
                raise ValueError(
                    f"{self.__class__.__name__}.get_metadata_files_with_dest() entry {idx} "
                    "dest_name must be a non-empty string"
                )
            if not isinstance(should_truncate, bool):
                raise ValueError(
                    f"{self.__class__.__name__}.get_metadata_files_with_dest() entry {idx} "
                    f"should_truncate must be bool, got {type(should_truncate).__name__}"
                )
            destination_to_sources.setdefault(dest_name, []).append(source_path)

        duplicate_destinations = {
            dest_name: sources
            for dest_name, sources in destination_to_sources.items()
            if len(sources) > 1
        }
        if duplicate_destinations:
            collision_lines = []
            for dest_name, sources in sorted(duplicate_destinations.items()):
                source_list = ", ".join(str(path) for path in sources)
                collision_lines.append(f"{dest_name}: {source_list}")
            collisions = "; ".join(collision_lines)
            raise ValueError(
                f"{self.__class__.__name__} metadata destination filenames must be unique: {collisions}"
            )

    def get_image_paths(self, camera: str = "left") -> List[Path]:
        """Get list of image file paths for the dataset.

        Returns paths respecting max_frames limit.

        Args:
            camera: Camera to get paths for ("left" or "right")

        Returns:
            List of Path objects to image files

        Raises:
            ValueError: If camera="right" is requested but no right camera paths exist
        """
        if camera not in {"left", "right"}:
            raise ValueError(f"Unsupported camera '{camera}'. Expected 'left' or 'right'")

        paths = []
        for frame_info in self._frames:
            if camera == "right":
                # Explicitly handle right camera - no silent fallback to left
                if 'image_path_right' in frame_info:
                    paths.append(Path(frame_info['image_path_right']))
                # Skip frames without right camera path (don't fallback to left)
            elif 'image_path' in frame_info:
                paths.append(Path(frame_info['image_path']))

        if camera == "right" and not paths and self._frames:
            raise ValueError(
                f"Right camera paths requested but none found in dataset. "
                f"Dataset may not have stereo data or 'image_path_right' key is missing. "
                f"Available keys in first frame: {list(self._frames[0].keys())}"
            )
        if camera == "left" and not paths and self._frames:
            raise ValueError(
                f"Left camera paths requested but none found in dataset. "
                f"Dataset may be malformed: required key 'image_path' is missing. "
                f"Available keys in first frame: {list(self._frames[0].keys())}"
            )

        return paths

    def create_truncated_copy(self, max_frames: int, output_dir: Optional[Path] = None) -> Path:
        """Create a truncated copy of this dataset with max_frames frames.

        This creates a new directory with hardlinks (or copies) of the first
        max_frames images and truncated metadata files. Useful for running
        SLAM algorithms on a subset of the dataset.

        Args:
            max_frames: Maximum number of frames to include
            output_dir: Optional output directory. If None, creates a temp directory.

        Returns:
            Path to the truncated dataset directory
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_truncated_copy(). "
            "This method must be implemented by dataset subclasses that support truncation."
        )

    def _link_or_copy_file(self, src: Path, dst: Path) -> None:
        """Create a hardlink or copy a file.

        Tries hardlink first (fast, no extra disk space), falls back to copy
        if hardlink fails (e.g., cross-filesystem).

        Args:
            src: Source file path
            dst: Destination file path
        """
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    def _truncate_text_file(
        self,
        src: Path,
        dst: Path,
        max_lines: int,
        preserve_comments: bool = True
    ) -> None:
        """Truncate a text file to max_lines data lines.

        Args:
            src: Source file path
            dst: Destination file path
            max_lines: Maximum number of data lines to keep
            preserve_comments: If True, preserve comment lines (starting with #)
        """
        copy_or_truncate_text_file(src, dst, max_lines, preserve_comments)

    def _symlink_image_dir(
        self,
        src_dir: Path,
        dst_dir: Path,
        max_frames: Optional[int] = None,
    ) -> None:
        """Create symlinks to images in source directory.

        Creates the destination directory and symlinks each image file.
        Respects max_frames limit if specified.

        Args:
            src_dir: Source directory containing images
            dst_dir: Destination directory (will be created)
            max_frames: Maximum number of images to symlink (None = all)
        """
        dst_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(src_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(src_dir.glob("*.jpg"))

        if max_frames is not None:
            image_files = image_files[:max_frames]

        for img_file in image_files:
            dst_file = dst_dir / img_file.name
            if not dst_file.exists():
                os.symlink(img_file, dst_file)

        logger.debug(f"Symlinked {len(image_files)} images: {src_dir} -> {dst_dir}")

    def get_ground_truth_path(self) -> Optional[Path]:
        """Get the path to the ground truth trajectory file.

        This method should be overridden by subclasses to return the dataset-specific
        ground truth file location.

        Returns:
            Path to ground truth file, or None if not available

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_ground_truth_path(). "
            "This method must be implemented by dataset subclasses."
        )

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Get metadata files with destination filenames.

        Some datasets have multiple files with the same name in different directories
        (e.g., EuRoC has cam0/data.csv and cam1/data.csv). This method allows specifying
        unique destination filenames to avoid overwriting.

        Returns:
            List of (source_path, dest_filename, should_truncate) tuples where:
            - source_path: Path to the source metadata file
            - dest_filename: Filename to use when copying (must be unique)
            - should_truncate: True if file should be truncated when copying with max_frames

        Returns:
            List of metadata tuples with explicit destination filenames.

        Raises:
            NotImplementedError: If subclass doesn't provide destination-aware metadata.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_metadata_files_with_dest(). "
            "This method must be implemented by dataset subclasses to provide explicit "
            "destination filenames for metadata copying."
        )

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Get the image directory name in the original dataset.

        Args:
            camera: Which camera - "left" or "right"

        Returns:
            Directory name containing images.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement get_image_directory_name(). "
            "This method must be implemented by dataset subclasses."
        )

    def _validate_camera_directory_name(self, camera: str, directory_name: str) -> str:
        """Validate and normalize dataset camera directory names."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
            )

        if not isinstance(directory_name, str):
            raise ValueError(
                f"{self.__class__.__name__}.get_image_directory_name('{camera}') must return str, "
                f"got {type(directory_name).__name__}"
            )

        normalized = directory_name.strip()
        if not normalized:
            raise ValueError(
                f"{self.__class__.__name__}.get_image_directory_name('{camera}') cannot return empty "
                "directory names. Use '.' when images are stored directly in the sequence root."
            )

        if Path(normalized).is_absolute():
            raise ValueError(
                f"{self.__class__.__name__}.get_image_directory_name('{camera}') must return a relative "
                f"path, got absolute path '{directory_name}'."
            )

        return normalized

    def resolve_camera_directory_name(self, source_root: Path, camera: str = "left") -> str:
        """Resolve a dataset camera directory name against a runtime source root.

        Args:
            source_root: Root path containing the current runtime image directories.
            camera: Camera role ("left" or "right").

        Returns:
            Normalized, dataset-native camera directory name.

        Raises:
            RuntimeError: If source root or resolved camera directory is missing.
            ValueError: If camera role or directory declaration is invalid.
        """
        source_root = Path(source_root)
        if not source_root.exists():
            raise RuntimeError(f"Camera source root does not exist: {source_root}")
        if not source_root.is_dir():
            raise RuntimeError(f"Camera source root is not a directory: {source_root}")

        directory_name = self._validate_camera_directory_name(
            camera,
            self.get_image_directory_name(camera),
        )
        resolved_dir = source_root / directory_name
        if not resolved_dir.exists() or not resolved_dir.is_dir():
            raise RuntimeError(
                f"Camera directory '{directory_name}' for role '{camera}' not found under {source_root}."
            )

        return directory_name

    def resolve_camera_directories(self, source_root: Path) -> Dict[str, str]:
        """Resolve active runtime camera directories for this dataset instance.

        Args:
            source_root: Root path containing runtime camera directories.

        Returns:
            Mapping from camera role to resolved dataset-native directory name.

        Raises:
            RuntimeError: If required camera directories are missing.
            ValueError: If dataset declares invalid camera roles or mappings.
        """
        camera_roles = self.get_active_camera_roles()
        if "left" not in camera_roles:
            raise RuntimeError(
                "Dataset must expose an active 'left' camera role for runtime camera resolution."
            )

        camera_dirs: Dict[str, str] = {}
        for role in camera_roles:
            camera_dirs[role] = self.resolve_camera_directory_name(source_root, role)

        if "right" in camera_dirs and camera_dirs["right"] == camera_dirs["left"]:
            raise RuntimeError(
                f"Resolved left and right cameras to the same directory: {camera_dirs['left']}"
            )

        return camera_dirs

    def get_canonical_camera_name(self, camera: str = "left") -> str:
        """Get the canonical output camera directory name.

        Canonical camera names are used for standardized perturbed output
        layout regardless of dataset-native directory conventions.

        Args:
            camera: Camera role ("left" or "right")

        Returns:
            Canonical directory name:
            - "image_2" for left camera
            - "image_3" for right camera

        Raises:
            ValueError: If camera role is not "left" or "right"
        """
        if camera == "left":
            return "image_2"
        if camera == "right":
            return "image_3"
        raise ValueError(
            f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
        )

    def get_depth_directory_path(self) -> Optional[Path]:
        """Get the path to native depth images directory if available.

        Returns:
            Path to depth directory, or None if dataset doesn't have native depth data.
        """
        return None  # Default: no depth directory

    def get_estimated_depth_directory(self, camera: str = "left") -> Optional[Path]:
        """Get the path to estimated depth maps directory if available.

        Estimated depth maps are generated by depth estimation backends and stored
        alongside the dataset root.

        Args:
            camera: Camera role ("left" or "right")

        Returns:
            Path to estimated depth directory if it exists, None otherwise.
        """
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
            )

        candidates = [
            self.path / f"{camera}_foundation_stereo_depth",
            self.path / f"{camera}_da3_depth",
            self.path / f"{camera}_depth",
        ]
        for depth_dir in candidates:
            if depth_dir.exists():
                return depth_dir
        return None

    def get_available_depth_dir(self, camera: str = "left") -> Optional[Path]:
        """Get reusable depth directory for a camera role, if available.

        This is the primary dataset API used by depth backend resolution. It should
        return a pre-existing depth directory aligned with the camera frames, or
        None when depth is unavailable and must be generated.

        Args:
            camera: Camera role ("left" or "right")

        Returns:
            Depth directory path or None.
        """
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
            )

        # Prefer native sensor depth when available.
        native_depth_dir = self.get_depth_directory_path()
        if native_depth_dir is not None:
            return native_depth_dir

        # Fall back to any existing estimated depth directory.
        return self.get_estimated_depth_directory(camera)

    def load_depth_for_frame(
        self,
        rgb_filename: str,
        camera: str = "left",
        use_estimated: bool = True
    ) -> Optional[np.ndarray]:
        """Load depth map for a specific frame.

        Tries to load depth from:
        1. Native depth directory (if available)
        2. Estimated depth directory (if use_estimated=True and available)

        Args:
            rgb_filename: RGB image filename (e.g., "000000.png")
            camera: Camera role ("left" or "right")
            use_estimated: Whether to use estimated depth if native isn't available

        Returns:
            Depth map as float32 array in meters, or None if not available.
        """
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
            )

        # Try native depth first
        native_depth_dir = self.get_depth_directory_path()
        if native_depth_dir is not None:
            depth_path = native_depth_dir / rgb_filename
            if depth_path.exists():
                return self._load_depth_file(depth_path, is_estimated=False)
            mapped_native_path = self._find_depth_path_from_frame_metadata(rgb_filename)
            if mapped_native_path is not None and mapped_native_path.exists():
                return self._load_depth_file(mapped_native_path, is_estimated=False)

        # Try estimated depth
        if use_estimated:
            estimated_candidates = [
                self.path / f"{camera}_foundation_stereo_depth",
                self.path / f"{camera}_da3_depth",
                self.path / f"{camera}_depth",
            ]
            for estimated_depth_dir in estimated_candidates:
                if not estimated_depth_dir.exists():
                    continue
                depth_path = estimated_depth_dir / rgb_filename
                if depth_path.exists():
                    return self._load_depth_file(depth_path, is_estimated=True)

        return None

    def _find_depth_path_from_frame_metadata(self, rgb_filename: str) -> Optional[Path]:
        """Find depth path from frame metadata using RGB filename association."""
        for frame_info in self._frames:
            image_path = frame_info.get("image_path")
            depth_path = frame_info.get("depth_path")
            if not image_path or not depth_path:
                continue
            if Path(image_path).name == rgb_filename:
                return Path(depth_path)
        return None

    def _load_depth_file(self, depth_path: Path, is_estimated: bool = False) -> np.ndarray:
        """Load a depth file and convert to meters.

        Args:
            depth_path: Path to depth PNG file
            is_estimated: If True, uses estimated depth encoding (value/256.0)
                         If False, uses native encoding (dataset-specific)

        Returns:
            Depth map as float32 array in meters.
        """
        import cv2

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"Failed to read depth map: {depth_path}")
        if depth.ndim != 2:
            raise RuntimeError(
                f"Invalid depth map at {depth_path}: expected single-channel image, "
                f"got shape {depth.shape}"
            )

        if is_estimated:
            # Estimated depth encoding: depth_meters = value / 256.0
            depth_meters = depth.astype(np.float32) / 256.0
        else:
            # Native depth - datasets own native decoding semantics.
            depth_meters = self.decode_native_depth(depth)

        return depth_meters

    def decode_native_depth(self, depth: np.ndarray) -> np.ndarray:
        """Decode native sensor depth values to meters.

        Dataset adapters should override this when native depth encoding differs
        from the default millimeter-based conversion.

        Args:
            depth: Raw native depth values read from disk.

        Returns:
            Depth in meters as float32.
        """
        # Default: assume native depth is stored in millimeters.
        return depth.astype(np.float32) / 1000.0

    def get_timestamps_file_path(self) -> Optional[Path]:
        """Get the path to the timestamps file.

        Returns:
            Path to timestamps file, or None if not available.
        """
        return None  # Default: no separate timestamps file

    def has_depth(self) -> bool:
        """Check if this dataset has native depth data.

        Returns:
            True if dataset includes depth maps, False otherwise.
        """
        return self.get_depth_directory_path() is not None

    def get_depth_source(self) -> str:
        """Determine the best available depth source for this dataset.

        Returns:
            One of: 'sensor', 'stereo', 'none'
        """
        if self.has_depth():
            return 'sensor'
        if "right" in self.get_active_camera_roles():
            return 'stereo'
        return 'none'

    def get_camera_intrinsics(self, camera: str = "left") -> Optional[CameraIntrinsics]:
        """Get per-camera intrinsic calibration.

        Args:
            camera: Camera role, either "left" or "right".

        Returns:
            CameraIntrinsics with fx/fy/cx/cy and optional width/height/baseline.
            Returns None when unavailable.

        Raises:
            ValueError: If camera is not "left" or "right".
        """
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )
        return None

    def requires_association_file(self) -> bool:
        """Check if dataset requires an RGB-depth association file for SLAM.

        Returns:
            True if dataset needs associations.txt mapping RGB to depth frames.
        """
        return False  # Default: no association file needed

    def create_pyslam_structure(
        self,
        images_path: Path,
        temp_root: Path,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Create the dataset layout PySLAM expects for this dataset type.

        Args:
            images_path: Path to images (perturbed output or original dataset path)
            temp_root: Temporary directory root to create the structure in
            max_frames: Maximum number of frames to include (None = all)

        Returns:
            Path to the created dataset structure root

        Raises:
            NotImplementedError: If subclass doesn't implement this method
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement create_pyslam_structure(). "
            "This method must be implemented to support PySLAM evaluation. "
            "See KITTI, TUM, or EuRoC implementations for examples."
        )

    def find_association_file(self) -> Optional[Path]:
        """Find the RGB-depth association file if one exists.

        Returns:
            Path to association file, or None if not found.
        """
        if not self.requires_association_file():
            return None

        # Common association file names in order of preference
        association_names = [
            "associations.txt",
            "associate.txt",
            "association.txt",
            "rgb_depth_associations.txt",
            "assoc.txt",
        ]

        for name in association_names:
            assoc_path = self.path / name
            if assoc_path.exists():
                return assoc_path

        return None

    def get_all_timestamps(self) -> Dict[int, float]:
        """Load all frame timestamps from loaded frame metadata.

        Returns:
            Dictionary mapping frame index to timestamp.
        """
        timestamps: Dict[int, float] = {}
        for idx, frame_info in enumerate(self._frames):
            if "timestamp" not in frame_info:
                raise ValueError(
                    f"{self.__class__.__name__} frame {idx} is missing required 'timestamp'."
                )
            ts_value = frame_info["timestamp"]
            if not isinstance(ts_value, (int, float)):
                raise ValueError(
                    f"{self.__class__.__name__} frame {idx} has non-numeric timestamp: {ts_value!r}"
                )
            timestamps[idx] = float(ts_value)
        return timestamps

    def get_algorithm_timestamps(self) -> Dict[int, float]:
        """Return frame-indexed timestamps used by SLAM wrappers.

        Dataset adapters can override this when wrappers need a dataset-native
        timestamp domain (for example, EuRoC nanoseconds).
        """
        return self.get_all_timestamps()

    def filter_metadata_for_kept_frames(
        self,
        output_dir: Path,
        kept_frame_indices: List[int],
        total_frames: int
    ) -> None:
        """Filter metadata files to only include frames that were kept (not dropped).

        When frame dropping is applied, the output dataset should only reference
        the frames that were actually saved. This method regenerates metadata files
        (like timestamps, associations) to only include kept frames.

        Args:
            output_dir: Directory where the perturbed output was saved
                       (e.g., results/experiment/images/module_name/)
            kept_frame_indices: List of frame indices that were kept (not dropped),
                               in the order they were processed
            total_frames: Total number of frames before dropping

        """
        if len(kept_frame_indices) >= total_frames:
            return

        raise NotImplementedError(
            f"{self.__class__.__name__} does not support dropped-frame metadata filtering. "
            "Implement filter_metadata_for_kept_frames() before using frame-drop perturbations "
            "with this dataset."
        )

    def __repr__(self) -> str:
        """String representation of the dataset."""
        return (
            f"{self.__class__.__name__}("
            f"path='{self.path}', "
            f"frames={len(self)}, "
            f"sequence='{self.sequence_name}')"
        )


class MockDataset(Dataset):
    """
    Mock dataset that generates synthetic data for testing.

    This dataset generates random images and optional depth maps
    without requiring any files on disk.
    """

    def __init__(
        self,
        config: DatasetConfig,
        num_frames: int = 100,
        image_size: Tuple[int, int] = (480, 640),
        with_depth: bool = True,
        with_ground_truth: bool = True
    ):
        """
        Initialize mock dataset.

        Args:
            config: Dataset configuration
            num_frames: Number of synthetic frames to generate
            image_size: Image dimensions (height, width)
            with_depth: Whether to include depth maps
            with_ground_truth: Whether to include ground truth trajectory
        """
        self.num_frames = num_frames
        self.image_size = image_size
        self.with_depth = with_depth
        self.with_ground_truth = with_ground_truth

        # Override max_frames if specified in config
        if config.max_frames:
            self.num_frames = min(self.num_frames, config.max_frames)

        super().__init__(config)

    def _validate_path(self) -> None:
        """Mock dataset doesn't need path validation."""
        pass

    def _load_dataset(self) -> None:
        """Generate synthetic frame metadata."""
        logger.debug(f"Generating {self.num_frames} mock frames")

        # Generate frame metadata
        for i in range(self.num_frames):
            self._frames.append({
                'index': i,
                'timestamp': float(i * 0.033),  # ~30 FPS
                'sequence_id': 'mock_sequence',
                'frame_id': i,
                'image_path': f"mock/frame_{i:06d}.png",
            })

        # Generate synthetic ground truth trajectory if requested
        if self.with_ground_truth:
            self._generate_ground_truth()

    def _generate_ground_truth(self) -> None:
        """Generate a synthetic circular trajectory."""
        num_poses = len(self._frames)
        self._ground_truth = np.zeros((num_poses, 4, 4))

        for i in range(num_poses):
            angle = 2 * np.pi * i / num_poses
            radius = 2.0

            # Translation
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            y = 0.5 * np.sin(4 * angle)  # Small vertical motion

            # Rotation (look towards center)
            yaw = angle + np.pi

            self._ground_truth[i] = np.array([
                [np.cos(yaw), 0, np.sin(yaw), x],
                [0, 1, 0, y],
                [-np.sin(yaw), 0, np.cos(yaw), z],
                [0, 0, 0, 1]
            ])

    def _load_frame(self, idx: int) -> Dict[str, Any]:
        """Generate synthetic frame data."""
        frame_info = self._frames[idx]

        # Generate random RGB image
        np.random.seed(idx)  # Reproducible random data
        image = np.random.randint(0, 256, (*self.image_size, 3), dtype=np.uint8)

        gradient = np.linspace(0, 1, self.image_size[1])
        image[:, :, 0] = (image[:, :, 0] * 0.7 + gradient * 255 * 0.3).astype(np.uint8)

        result = {
            'image': image,
            'timestamp': frame_info['timestamp'],
            'sequence_id': frame_info['sequence_id'],
            'frame_id': frame_info['frame_id']
        }

        # Generate depth map if requested
        if self.with_depth:
            depth = np.ones(self.image_size, dtype=np.float32) * 5.0
            cx, cy = self.image_size[1] // 2, self.image_size[0] // 2
            y_coords, x_coords = np.ogrid[:self.image_size[0], :self.image_size[1]]
            dist_from_center = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
            depth = 2.0 + (dist_from_center / (self.image_size[1] / 2)) * 3.0
            depth = depth.astype(np.float32)
            result['depth'] = depth
        else:
            result['depth'] = None

        return result

    def get_ground_truth_path(self) -> Optional[Path]:
        """Mock dataset has no ground truth file."""
        return None

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Mock dataset has no metadata files."""
        return []

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Mock dataset images are exposed from the dataset root."""
        return "."
