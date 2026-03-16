"""7-Scenes RGB-D dataset loader.

Microsoft Research 7-Scenes dataset for indoor RGB-D SLAM and relocalization.
https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from ..utils.paths import create_temp_dir
from ..utils import get_logger, read_image, read_depth
from ..config.schema import DatasetConfig
from .base import CameraIntrinsics, Dataset
from .catalog import get_7scenes_sequence, list_7scenes_sequences, get_7scenes_base_dir
from .download import ensure_7scenes_sequence

logger = get_logger(__name__)


class SevenScenesDataset(Dataset):
    """
    Loader for Microsoft 7-Scenes RGB-D dataset.

    The 7-Scenes dataset format consists of:
    - Color images: frame-XXXXXX.color.png (RGB, 24-bit PNG)
    - Depth images: frame-XXXXXX.depth.png (16-bit PNG, millimeters, 65535=invalid)
    - Camera poses: frame-XXXXXX.pose.txt (4x4 camera-to-world matrix)

    Directory structure:
        7scenes/
        ├── chess/
        │   ├── seq-01/
        │   │   ├── frame-000000.color.png
        │   │   ├── frame-000000.depth.png
        │   │   ├── frame-000000.pose.txt
        │   │   └── ...
        │   ├── seq-02/
        │   └── ...
        ├── fire/
        └── ...

    Supports sequence-based configuration:
        dataset:
          type: 7scenes
          sequence: chess/seq-01
          # or just scene name to load all sequences:
          sequence: chess
    """

    # Depth scale: 7-Scenes uses millimeters, 65535 = invalid
    DEPTH_SCALE = 1000.0  # Convert mm to meters
    INVALID_DEPTH = 65535

    # Default camera intrinsics (from Microsoft documentation)
    DEFAULT_INTRINSICS = {
        'fx': 585.0,
        'fy': 585.0,
        'cx': 320.0,
        'cy': 240.0,
        'width': 640,
        'height': 480,
    }

    @property
    def supports_stereo(self) -> bool:
        """7-Scenes uses a single RGB-D camera (no stereo pair)."""
        return False

    def _sequence_name_from_path(self) -> str:
        """Infer canonical 7-Scenes sequence name from resolved path."""
        if self._is_sequence_dir(self.path):
            return f"{self.path.parent.name}/{self.path.name}"
        return self.path.name

    def _normalize_sequence_for_compare(self, sequence: str) -> str:
        """Normalize 7-Scenes sequence labels for config/path consistency checks."""
        normalized = super()._normalize_sequence_for_compare(sequence)
        normalized = normalized.replace("\\", "/").strip("/")
        return normalized.lower()

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        """
        Resolve 7-Scenes dataset path, downloading if necessary.

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
                "7-Scenes dataset requires either 'path' or 'sequence'. "
                "Example sequences: chess, chess/seq-01, fire/seq-02"
            )

        entry = get_7scenes_sequence(config.sequence)
        if entry is None:
            available = list_7scenes_sequences()
            raise ValueError(
                f"Unknown 7-Scenes scene '{config.sequence}'.\n"
                f"Available scenes: {', '.join(sorted(available))}"
            )

        # ensure_7scenes_sequence handles download if needed
        sequence_path = ensure_7scenes_sequence(config.sequence)
        logger.info(f"Resolved 7-Scenes sequence '{config.sequence}' to: {sequence_path}")
        return str(sequence_path)

    def __init__(self, config: DatasetConfig):
        """
        Initialize 7-Scenes dataset.

        Args:
            config: Dataset configuration
        """
        super().__init__(config)

    def _load_dataset(self) -> None:
        """Load 7-Scenes dataset by scanning for frame files."""
        logger.info(f"Loading 7-Scenes dataset from {self.path}")

        if self._is_sequence_dir(self.path):
            # Single sequence
            sequences = [('', self.path)]
            scene_name = self.path.parent.name
        else:
            # Scene directory - load all sequences
            scene_name = self.path.name
            sequences = self._find_sequences(self.path)

        if not sequences:
            raise FileNotFoundError(
                f"No sequences found in {self.path}\n"
                f"Expected structure: {self.path}/seq-XX/frame-XXXXXX.color.png"
            )

        for seq_name, seq_path in sequences:
            self._load_sequence(seq_path, scene_name, seq_name)

        self._load_ground_truth_from_poses()

        logger.info(f"Loaded {len(self._frames)} frames from 7-Scenes {scene_name}")

    def _is_sequence_dir(self, path: Path) -> bool:
        """Check if path is a sequence directory (contains frame files)."""
        return any(path.glob('frame-*.color.png'))

    def _find_sequences(self, scene_path: Path) -> List[Tuple[str, Path]]:
        """Find all sequence directories in a scene."""
        sequences = []
        for seq_dir in sorted(scene_path.iterdir()):
            if seq_dir.is_dir() and seq_dir.name.startswith('seq-'):
                if self._is_sequence_dir(seq_dir):
                    sequences.append((seq_dir.name, seq_dir))
        return sequences

    def _load_sequence(self, seq_path: Path, scene_name: str, seq_name: str) -> None:
        """Load frames from a single sequence."""
        color_files = sorted(seq_path.glob('frame-*.color.png'))

        if not color_files:
            logger.warning(f"No frames found in {seq_path}")
            return

        sequence_id = f"{scene_name}/{seq_name}" if seq_name else scene_name

        for idx, color_path in enumerate(color_files):
            # Extract frame number from filename
            frame_name = color_path.stem.replace('.color', '')
            frame_num = int(frame_name.split('-')[1])

            depth_path = seq_path / f"{frame_name}.depth.png"
            pose_path = seq_path / f"{frame_name}.pose.txt"

            frame_info = {
                'image_path': str(color_path),
                'timestamp': float(frame_num) / 30.0,  # Assume 30 FPS
                'sequence_id': sequence_id,
                'frame_id': len(self._frames),
                'frame_num': frame_num,
                'filename': color_path.name,
            }
            if depth_path.exists():
                frame_info['depth_path'] = str(depth_path)
            if pose_path.exists():
                frame_info['pose_path'] = str(pose_path)
            self._frames.append(frame_info)

    def _load_ground_truth_from_poses(self) -> None:
        """Load ground truth trajectory from individual pose files."""
        poses = []

        for frame_info in self._frames:
            pose_path = frame_info.get('pose_path')
            if pose_path and Path(pose_path).exists():
                try:
                    pose = self._load_pose_file(pose_path)
                    poses.append(pose)
                except Exception as e:
                    logger.warning(f"Failed to load pose {pose_path}: {e}")
                    poses.append(np.eye(4))
            else:
                poses.append(np.eye(4))

        if poses:
            self._ground_truth = np.array(poses)
            logger.info(f"Loaded {len(poses)} ground truth poses")

    def _load_pose_file(self, pose_path: str) -> np.ndarray:
        """Load 4x4 pose matrix from text file."""
        with open(pose_path, 'r') as f:
            lines = f.readlines()

        matrix = []
        for line in lines:
            line = line.strip()
            if line:
                values = [float(v) for v in line.split()]
                if len(values) == 4:
                    matrix.append(values)

        if len(matrix) != 4:
            raise ValueError(f"Invalid pose file format: expected 4x4 matrix")

        return np.array(matrix)

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
                f"7-Scenes frame {idx}: failed to load color image '{image_path}': {e}"
            ) from e

        depth = None
        if frame_info.get('depth_path'):
            try:
                depth = read_depth(
                    frame_info['depth_path'],
                    scale=self.DEPTH_SCALE
                )
                # Mark invalid depth values as 0
                depth[depth >= (self.INVALID_DEPTH / self.DEPTH_SCALE)] = 0
            except Exception as e:
                logger.error(f"Error loading depth: {e}")
                depth = None

        return {
            'image': image,
            'depth': depth,
            'timestamp': frame_info['timestamp'],
            'sequence_id': frame_info['sequence_id'],
            'frame_id': frame_info['frame_id'],
            'rgb_filename': frame_info['filename'],
        }

    def get_ground_truth_path(self) -> Optional[Path]:
        """Get path to ground truth.

        7-Scenes stores poses per-frame, not in a single file.
        Returns None - use get_ground_truth() for the trajectory array.

        Returns:
            None (poses are stored per-frame, not in a single file)
        """
        # 7-Scenes doesn't have a single GT file, poses are per-frame
        return None

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Get 7-Scenes metadata files with explicit destination filenames.

        Uses directory-prefixed destination names to avoid collisions when
        both scene-level and parent-level split files are present.
        """
        metadata_files_with_dest = []

        for split_file in ["TrainSplit.txt", "TestSplit.txt"]:
            split_path = self.path / split_file
            if split_path.exists():
                metadata_files_with_dest.append(
                    (split_path, f"{split_path.parent.name}_{split_path.name}", False)
                )

            if self.path.parent.exists():
                parent_split = self.path.parent / split_file
                if parent_split.exists():
                    metadata_files_with_dest.append(
                        (parent_split, f"{parent_split.parent.name}_{parent_split.name}", False)
                    )

        return metadata_files_with_dest

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Get the 7-Scenes image directory name.

        7-Scenes stores images directly in sequence folders, not subdirectories.

        Args:
            camera: Only "left" is supported (single RGB-D camera)

        Returns:
            "." to indicate images are in the sequence root
        """
        if camera == "right":
            raise ValueError(
                "7-Scenes is single-camera RGB-D; right camera directory is unavailable"
            )
        # 7-Scenes stores images directly in sequence folder
        return "."

    def get_depth_directory_path(self) -> Optional[Path]:
        """Get the path to depth images directory.

        7-Scenes stores depth alongside color in the same directory.

        Returns:
            Path to the sequence directory containing depth files
        """
        # Depth files are in the same directory as color files
        if self._is_sequence_dir(self.path):
            return self.path
        return None

    def requires_association_file(self) -> bool:
        """7-Scenes doesn't require an association file.

        Color and depth are paired by frame number in filename.

        Returns:
            False - files are paired by naming convention
        """
        return False

    def get_camera_intrinsics(self, camera: str = "left") -> Optional[CameraIntrinsics]:
        """Get default camera intrinsics for 7-Scenes."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )

        if camera == "right":
            return None

        return CameraIntrinsics(
            fx=float(self.DEFAULT_INTRINSICS["fx"]),
            fy=float(self.DEFAULT_INTRINSICS["fy"]),
            cx=float(self.DEFAULT_INTRINSICS["cx"]),
            cy=float(self.DEFAULT_INTRINSICS["cy"]),
            width=float(self.DEFAULT_INTRINSICS["width"]),
            height=float(self.DEFAULT_INTRINSICS["height"]),
        )

    def decode_native_depth(self, depth: np.ndarray) -> np.ndarray:
        """Convert 7-Scenes native depth values to meters.

        7-Scenes uses 16-bit PNG with millimeter scale, 65535 = invalid.

        Args:
            depth: Raw depth values from 16-bit PNG file

        Returns:
            Depth in meters as float32, with invalid values set to 0
        """
        depth_meters = depth.astype(np.float32) / self.DEPTH_SCALE
        depth_meters[depth >= self.INVALID_DEPTH] = 0
        return depth_meters

    def create_truncated_copy(self, max_frames: int, output_dir: Optional[Path] = None) -> Path:
        """Create a truncated copy of this 7-Scenes dataset.

        Creates hardlinks to the first max_frames images and their associated files.

        Args:
            max_frames: Maximum number of frames to include
            output_dir: Optional output directory. If None, creates a temp directory.

        Returns:
            Path to the truncated dataset directory
        """
        if output_dir is None:
            output_dir = create_temp_dir(prefix="7scenes_truncated_")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        frames_to_copy = self._frames[:max_frames]

        for frame_info in frames_to_copy:
            color_path = Path(frame_info['image_path'])
            rel_path = color_path.relative_to(self.path) if self._is_sequence_dir(self.path) else color_path.name

            dst_color = output_dir / rel_path
            dst_color.parent.mkdir(parents=True, exist_ok=True)

            self._link_or_copy_file(color_path, dst_color)

            if frame_info.get('depth_path'):
                depth_path = Path(frame_info['depth_path'])
                dst_depth = dst_color.parent / depth_path.name
                if depth_path.exists():
                    self._link_or_copy_file(depth_path, dst_depth)

            if frame_info.get('pose_path'):
                pose_path = Path(frame_info['pose_path'])
                dst_pose = dst_color.parent / pose_path.name
                if pose_path.exists():
                    self._link_or_copy_file(pose_path, dst_pose)

        for meta_path, dest_name, _should_truncate in self.get_metadata_files_with_dest():
            if meta_path.exists():
                dst_meta = output_dir / dest_name
                self._link_or_copy_file(meta_path, dst_meta)

        logger.info(f"Created truncated 7-Scenes dataset ({max_frames} frames) at {output_dir}")
        return output_dir

    def export_ground_truth_tum(self, output_path: Path) -> None:
        """Export ground truth trajectory to TUM format.

        7-Scenes stores poses per-frame. This method exports them to a single
        TUM-format file for compatibility with evaluation tools.

        Args:
            output_path: Path to output TUM trajectory file
        """
        if self._ground_truth is None:
            logger.warning("No ground truth available to export")
            return

        from scipy.spatial.transform import Rotation

        with open(output_path, 'w') as f:
            f.write("# ground truth trajectory (TUM format)\n")
            f.write("# timestamp tx ty tz qx qy qz qw\n")

            for idx, pose in enumerate(self._ground_truth):
                timestamp = self._frames[idx]['timestamp']
                tx, ty, tz = pose[:3, 3]
                quat = Rotation.from_matrix(pose[:3, :3]).as_quat()  # x, y, z, w
                qx, qy, qz, qw = quat

                f.write(f"{timestamp:.6f} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        logger.info(f"Exported ground truth to TUM format: {output_path}")
