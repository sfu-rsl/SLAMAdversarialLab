"""TUM RGB-D dataset loader."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..utils import get_logger, read_image, read_depth
from ..utils.paths import create_temp_dir
from ..config.schema import DatasetConfig
from .base import CameraIntrinsics, Dataset
from .catalog import get_tum_sequence, list_tum_sequences, get_tum_base_dir
from .download import ensure_tum_sequence

logger = get_logger(__name__)


class TUMDataset(Dataset):
    """
    Loader for TUM RGB-D dataset format.

    The TUM RGB-D dataset format consists of:
    - RGB images in rgb/ directory (PNG format)
    - Depth images in depth/ directory (PNG format, 16-bit)
    - associations.txt or assoc.txt: Timestamps and file mappings
    - groundtruth.txt: Ground truth trajectory (optional)

    Depth encoding: depth_meters = depth_png_value / 5000.0

    Supports sequence-based configuration with auto-download:
        dataset:
          type: tum
          sequence: freiburg1_desk  # Auto-downloads if not present
    """

    DEPTH_SCALE = 5000.0  # TUM uses 5000 to represent 1 meter

    TUM_INTRINSICS_BY_CATEGORY = {
        "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3, "width": 640.0, "height": 480.0},
        "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7, "width": 640.0, "height": 480.0},
        "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6, "width": 640.0, "height": 480.0},
    }
    DEFAULT_INTRINSICS = TUM_INTRINSICS_BY_CATEGORY["freiburg1"]

    @property
    def supports_stereo(self) -> bool:
        """TUM RGB-D sequences are monocular (single RGB camera + depth)."""
        return False

    def _normalize_sequence_for_compare(self, sequence: str) -> str:
        """Normalize TUM sequence labels for config/path consistency checks."""
        normalized = super()._normalize_sequence_for_compare(sequence).lower()
        if normalized.startswith("rgbd_dataset_"):
            normalized = normalized[len("rgbd_dataset_"):]
        if normalized.startswith("tum_"):
            normalized = normalized[len("tum_"):]
        return normalized

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        """
        Resolve TUM dataset path, downloading if necessary.

        Args:
            config: Dataset configuration with either path or sequence

        Returns:
            Resolved path string

        Raises:
            ValueError: If sequence is unknown or path cannot be resolved
        """
        # If path is provided, use it directly
        if config.path:
            return config.path

        # Must have sequence for auto-resolution
        if not config.sequence:
            raise ValueError(
                "TUM dataset requires either 'path' or 'sequence'. "
                "Example sequences: freiburg1_desk, freiburg1_xyz, freiburg2_desk"
            )

        entry = get_tum_sequence(config.sequence)
        if entry is None:
            available = list_tum_sequences()
            raise ValueError(
                f"Unknown TUM sequence '{config.sequence}'.\n"
                f"Available sequences: {', '.join(sorted(available))}"
            )

        # ensure_tum_sequence handles download if needed
        sequence_path = ensure_tum_sequence(config.sequence)
        logger.info(f"Resolved TUM sequence '{config.sequence}' to: {sequence_path}")
        return str(sequence_path)

    def __init__(self, config: DatasetConfig):
        """
        Initialize TUM dataset.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load TUM dataset metadata and file associations."""
        logger.info(f"Loading TUM dataset from {self.path}")

        associations = self._load_associations()

        if not associations:
            raise RuntimeError(
                f"Could not load frame associations from {self.path}. "
                "TUM datasets require an association file (rgb.csv, associations.txt, "
                "assoc.txt, associate.txt, or rgb.txt)."
            )

        for assoc in associations:
            frame_info = {
                'timestamp': assoc['timestamp'],
                'image_path': assoc['image_path'],
                'sequence_id': assoc.get('sequence_id', self.sequence_name),
                'frame_id': assoc.get('frame_id', 0)
            }
            if assoc.get('depth_path'):
                frame_info['depth_path'] = assoc['depth_path']
            self._frames.append(frame_info)

        self._load_ground_truth()

        logger.info(f"Loaded {len(self._frames)} frames from TUM dataset")

    def _load_associations(self) -> List[Dict[str, Any]]:
        """
        Load associations from file.

        Returns:
            List of association dictionaries
        """
        associations = []

        # Try different possible association file names
        # Prioritize rgb.csv (VSLAM-LAB format) for better compatibility
        assoc_files = [
            'rgb.csv',  # VSLAM-LAB format (has all frames)
            'associations.txt',
            'assoc.txt',
            'associate.txt',
            'rgb.txt'  # Some datasets use this
        ]

        assoc_path = None
        for filename in assoc_files:
            candidate = self.path / filename
            if candidate.exists():
                assoc_path = candidate
                break

        if not assoc_path:
            logger.debug("No associations file found")
            return []

        logger.info(f"Loading associations from {assoc_path}")

        try:
            is_csv = str(assoc_path).endswith('.csv')

            with open(assoc_path, 'r') as f:
                frame_id = 0
                for line_num, line in enumerate(f):
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Skip CSV header
                    if is_csv and line_num == 0 and 'ts_rgb' in line:
                        continue

                    if is_csv:
                        # VSLAM-LAB CSV format:
                        # ts_rgb0 (s),path_rgb0,ts_depth0 (s),path_depth0
                        parts = line.split(',')
                        if len(parts) >= 2:
                            timestamp = float(parts[0])
                            rgb_file = parts[1]

                            # Extract just the filename from rgb_0/filename.png
                            # Support both 'rgb' and 'rgb_0' folder naming
                            rgb_basename = Path(rgb_file).name
                            rgb_path = self.path / 'rgb' / rgb_basename
                            if not rgb_path.exists():
                                rgb_path = self.path / 'rgb_0' / rgb_basename

                            assoc = {
                                'timestamp': timestamp,
                                'image_path': str(rgb_path),
                                'frame_id': frame_id
                            }

                            if len(parts) >= 4:
                                depth_file = parts[3]
                                depth_basename = Path(depth_file).name
                                depth_path = self.path / 'depth' / depth_basename
                                if not depth_path.exists():
                                    depth_path = self.path / 'depth_0' / depth_basename
                                assoc['depth_path'] = str(depth_path)
                                assoc['depth_timestamp'] = float(parts[2])

                            associations.append(assoc)
                            frame_id += 1
                    else:
                        parts = line.split()

                        if len(parts) >= 2:
                            timestamp = float(parts[0])
                            rgb_file = parts[1]

                            rgb_path = self.path / rgb_file

                            assoc = {
                                'timestamp': timestamp,
                                'image_path': str(rgb_path),
                                'frame_id': frame_id
                            }

                            if len(parts) >= 4:
                                depth_file = parts[3]
                                depth_path = self.path / depth_file
                                assoc['depth_path'] = str(depth_path)
                                assoc['depth_timestamp'] = float(parts[2])

                            associations.append(assoc)
                            frame_id += 1

        except Exception as e:
            logger.error(f"Error loading associations: {e}")
            return []

        return associations

    def _load_ground_truth(self) -> None:
        """Load ground truth trajectory if available."""
        gt_files = [
            'groundtruth.txt',
            'gt.txt',
            'trajectory.txt'
        ]

        gt_path = None
        for filename in gt_files:
            candidate = self.path / filename
            if candidate.exists():
                gt_path = candidate
                break

        if not gt_path:
            logger.debug("No ground truth file found")
            return

        logger.info(f"Loading ground truth from {gt_path}")

        try:
            # Read ground truth file
            timestamps = []
            positions = []
            quaternions = []

            with open(gt_path, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split()

                    # Expected format:
                    # timestamp tx ty tz qx qy qz qw
                    if len(parts) >= 8:
                        timestamps.append(float(parts[0]))
                        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
                        quaternions.append([float(parts[4]), float(parts[5]),
                                          float(parts[6]), float(parts[7])])

            if timestamps:
                self._ground_truth = self._quaternions_to_matrices(positions, quaternions)
                logger.info(f"Loaded {len(self._ground_truth)} ground truth poses")

        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")

    def _quaternions_to_matrices(self, positions: List[List[float]],
                                quaternions: List[List[float]]) -> np.ndarray:
        """
        Convert position + quaternion to 4x4 transformation matrices.

        Args:
            positions: List of [x, y, z] positions
            quaternions: List of [qx, qy, qz, qw] quaternions

        Returns:
            Array of 4x4 transformation matrices
        """
        n = len(positions)
        matrices = np.zeros((n, 4, 4))

        for i in range(n):
            tx, ty, tz = positions[i]

            qx, qy, qz, qw = quaternions[i]

            norm = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
            if norm > 0:
                qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm

            R = np.array([
                [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
                [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
                [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)]
            ])

            matrices[i, :3, :3] = R
            matrices[i, :3, 3] = [tx, ty, tz]
            matrices[i, 3, 3] = 1.0

        return matrices

    def _load_frame(self, idx: int) -> Dict[str, Any]:
        """
        Load a specific frame's data.

        Args:
            idx: Frame index

        Returns:
            Frame data dictionary
        """
        frame_info = self._frames[idx]

        image_path = frame_info['image_path']
        try:
            image = read_image(image_path)
        except Exception as e:
            raise RuntimeError(
                f"TUM frame {idx}: failed to load RGB image '{image_path}': {e}"
            ) from e

        depth = None
        depth_filename = None
        if frame_info.get('depth_path'):
            depth_path = frame_info['depth_path']
            depth_filename = Path(depth_path).name
            try:
                depth = read_depth(depth_path, scale=self.DEPTH_SCALE)
            except Exception as e:
                logger.error(f"Error loading depth image {depth_path}: {e}")
                depth = None

        # Extract original filename from image path for output preservation
        rgb_filename = Path(frame_info['image_path']).name

        return {
            'image': image,
            'depth': depth,
            'timestamp': frame_info['timestamp'],
            'sequence_id': frame_info['sequence_id'],
            'frame_id': frame_info['frame_id'],
            'rgb_filename': rgb_filename,
            'depth_filename': depth_filename,
        }

    def get_ground_truth_path(self) -> Optional[Path]:
        """Get the path to the TUM ground truth trajectory file.

        TUM ground truth is stored directly in the dataset directory as groundtruth.txt.

        Returns:
            Path to ground truth file, or None if not available
        """
        # Try common ground truth file names
        gt_files = ['groundtruth.txt', 'gt.txt', 'trajectory.txt']

        for filename in gt_files:
            gt_path = self.path / filename
            if gt_path.exists():
                return gt_path

        logger.debug(f"TUM ground truth not found in {self.path}")
        return None

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Get list of TUM metadata files with destination filenames.

        Returns:
            List of (source_path, dest_name, should_truncate) tuples:
            - associations.txt, rgb.txt, depth.txt, rgb.csv (truncate)
            - groundtruth.txt (no truncate)
        """
        metadata_files = []

        truncate_files = ["associations.txt", "rgb.txt", "depth.txt", "rgb.csv"]
        for filename in truncate_files:
            file_path = self.path / filename
            if file_path.exists():
                metadata_files.append((file_path, filename, True))

        gt_path = self.get_ground_truth_path()
        if gt_path:
            metadata_files.append((gt_path, gt_path.name, False))

        return metadata_files

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Get the TUM RGB image directory name.

        Args:
            camera: Only "left" is supported (TUM is monocular RGB-D)

        Returns:
            "rgb" - TUM uses rgb/ directory for color images
        """
        if camera == "right":
            raise ValueError(
                "TUM RGB-D dataset is monocular; right camera directory is unavailable"
            )
        return "rgb"

    def _infer_intrinsics_category(self) -> str:
        """Infer Freiburg camera category from sequence or resolved path name."""
        candidates = [
            (self.config.sequence or "").lower(),
            self.path.name.lower(),
        ]

        for candidate in candidates:
            if "freiburg1" in candidate:
                return "freiburg1"
            if "freiburg2" in candidate:
                return "freiburg2"
            if "freiburg3" in candidate:
                return "freiburg3"

        return "freiburg1"

    def get_camera_intrinsics(self, camera: str = "left") -> Optional[CameraIntrinsics]:
        """Get TUM intrinsics for the requested camera."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )

        if camera == "right":
            return None

        category = self._infer_intrinsics_category()
        intrinsics = self.TUM_INTRINSICS_BY_CATEGORY.get(category, self.DEFAULT_INTRINSICS)
        return CameraIntrinsics(
            fx=float(intrinsics["fx"]),
            fy=float(intrinsics["fy"]),
            cx=float(intrinsics["cx"]),
            cy=float(intrinsics["cy"]),
            width=float(intrinsics["width"]) if "width" in intrinsics else None,
            height=float(intrinsics["height"]) if "height" in intrinsics else None,
        )

    def get_depth_directory_path(self) -> Optional[Path]:
        """Get the path to TUM depth images directory.

        Returns:
            Path to depth/ directory (resolved if symlink), or None if not found.
        """
        depth_path = self.path / "depth"
        if depth_path.exists():
            return depth_path.resolve() if depth_path.is_symlink() else depth_path
        return None

    def requires_association_file(self) -> bool:
        """TUM RGB-D requires association file mapping RGB to depth frames.

        Returns:
            True - TUM always needs RGB-depth timestamp association
        """
        return True

    def decode_native_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convert TUM native depth values to meters.

        TUM uses 16-bit PNG with scale factor of 5000 (5000 = 1 meter).

        Args:
            depth: Raw depth values from 16-bit PNG file

        Returns:
            Depth in meters as float32
        """
        return depth.astype(np.float32) / self.DEPTH_SCALE

    def create_pyslam_structure(
        self,
        images_path: Path,
        temp_root: Path,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Create TUM structure for PySLAM.

        PySLAM expects:
            base_path/{seq_name}/rgb/*.png
            base_path/{seq_name}/depth/*.png (for RGBD)
            base_path/{seq_name}/associations.txt
            base_path/{seq_name}/groundtruth.txt

        Args:
            images_path: Path to images (perturbed output or original dataset)
            temp_root: Temporary directory root to create structure in
            max_frames: Maximum number of frames to include

        Returns:
            Path to the created dataset structure root
        """
        sequence_name = self.sequence_name
        seq_dir = temp_root / sequence_name
        seq_dir.mkdir(parents=True, exist_ok=True)
        rgb_dir_name = self.get_image_directory_name("left")
        canonical_left_dir_name = self.get_canonical_camera_name("left")

        def _has_images(directory: Path) -> bool:
            if not directory.exists() or not directory.is_dir():
                return False
            return any(directory.glob("*.png")) or any(directory.glob("*.jpg"))

        rgb_candidate = images_path / rgb_dir_name
        canonical_candidate = images_path / canonical_left_dir_name

        if _has_images(rgb_candidate):
            rgb_images_dir = rgb_candidate
        elif _has_images(canonical_candidate):
            rgb_images_dir = canonical_candidate
        elif _has_images(images_path):
            # Some callers may pass a flat image directory directly.
            rgb_images_dir = images_path
        else:
            raise RuntimeError(
                f"Could not resolve TUM RGB image directory under {images_path}. "
                f"Checked: {rgb_candidate}, {canonical_candidate}, {images_path}"
            )

        # Symlink RGB images
        self._symlink_image_dir(rgb_images_dir, seq_dir / "rgb", max_frames)

        depth_path = self.get_depth_directory_path()
        if depth_path and depth_path.exists():
            self._symlink_image_dir(depth_path, seq_dir / "depth", max_frames)

        for file_path, dest_name, should_truncate in self.get_metadata_files_with_dest():
            if file_path.exists():
                if should_truncate and max_frames:
                    self._truncate_text_file(
                        file_path, seq_dir / dest_name, max_frames, preserve_comments=True
                    )
                else:
                    shutil.copy(file_path, seq_dir / dest_name)
                logger.debug(f"{'Truncated' if should_truncate else 'Copied'} {dest_name}")

        logger.info(f"Created TUM pyslam structure at {temp_root}")
        return temp_root

    def create_truncated_copy(self, max_frames: int, output_dir: Optional[Path] = None) -> Path:
        """Create a truncated copy of this TUM dataset.

        Creates hardlinks to the first max_frames images and truncates metadata files.

        Args:
            max_frames: Maximum number of frames to include
            output_dir: Optional output directory. If None, creates a temp directory.

        Returns:
            Path to the truncated dataset directory
        """
        if output_dir is None:
            output_dir = create_temp_dir(prefix="tum_truncated_")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        def _iter_data_lines(file_path: Path):
            with open(file_path, "r", encoding="utf-8") as file_handle:
                for raw_line in file_handle:
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    yield line

        def _collect_paths_from_associations(file_path: Path, limit: int) -> List[Path]:
            relative_paths: List[Path] = []
            for line in _iter_data_lines(file_path):
                parts = line.split()
                if len(parts) >= 2:
                    relative_paths.append(Path(parts[1]))
                if len(parts) >= 4:
                    relative_paths.append(Path(parts[3]))
                if len(relative_paths) >= (limit * 2):
                    break
            return relative_paths

        def _collect_paths_from_timestamp_list(file_path: Path, limit: int) -> List[Path]:
            relative_paths: List[Path] = []
            for line in _iter_data_lines(file_path):
                parts = line.split()
                if len(parts) >= 2:
                    relative_paths.append(Path(parts[1]))
                if len(relative_paths) >= limit:
                    break
            return relative_paths

        def _link_relative_paths(relative_paths: List[Path]) -> bool:
            if not relative_paths:
                return False

            linked_any = False
            seen = set()
            for relative_path in relative_paths:
                key = relative_path.as_posix()
                if key in seen:
                    continue
                seen.add(key)

                if relative_path.is_absolute():
                    raise RuntimeError(
                        f"TUM metadata path must be relative, got absolute path: {relative_path}"
                    )

                src_path = self.path / relative_path
                if not src_path.exists():
                    raise FileNotFoundError(
                        f"TUM truncated copy expected metadata-referenced file at {src_path}, but it does not exist"
                    )

                dst_path = output_dir / relative_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                self._link_or_copy_file(src_path, dst_path)
                linked_any = True

            return linked_any

        linked_from_metadata = False
        assoc_file = self.path / "associations.txt"
        if assoc_file.exists():
            linked_from_metadata = _link_relative_paths(
                _collect_paths_from_associations(assoc_file, max_frames)
            )

        if not linked_from_metadata:
            rgb_list = self.path / "rgb.txt"
            depth_list = self.path / "depth.txt"
            timestamp_list_paths: List[Path] = []
            if rgb_list.exists():
                timestamp_list_paths.extend(_collect_paths_from_timestamp_list(rgb_list, max_frames))
            if depth_list.exists():
                timestamp_list_paths.extend(_collect_paths_from_timestamp_list(depth_list, max_frames))
            linked_from_metadata = _link_relative_paths(timestamp_list_paths)

        if not linked_from_metadata:
            # Final fallback for minimal datasets missing metadata lists.
            for img_dir in ["rgb", "depth"]:
                src_dir = self.path / img_dir
                if not src_dir.exists():
                    continue

                dst_dir = output_dir / img_dir
                dst_dir.mkdir(parents=True, exist_ok=True)

                img_files = sorted(src_dir.glob("*.png"))[:max_frames]
                if not img_files:
                    img_files = sorted(src_dir.glob("*.jpg"))[:max_frames]

                for img_file in img_files:
                    self._link_or_copy_file(img_file, dst_dir / img_file.name)

        # Truncate metadata files (preserve comment lines)
        for txt_file in ["associations.txt", "rgb.txt", "depth.txt"]:
            src_file = self.path / txt_file
            if src_file.exists():
                self._truncate_text_file(
                    src_file,
                    output_dir / txt_file,
                    max_frames,
                    preserve_comments=True
                )

        gt_file = self.path / "groundtruth.txt"
        if gt_file.exists():
            self._link_or_copy_file(gt_file, output_dir / "groundtruth.txt")

        logger.info(f"Created truncated TUM dataset ({max_frames} frames) at {output_dir}")
        return output_dir
