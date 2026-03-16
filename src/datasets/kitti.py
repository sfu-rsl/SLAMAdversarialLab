"""KITTI stereo dataset loader."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from ..utils import get_logger, read_image, parse_kitti_calibration
from ..utils.paths import create_temp_dir
from ..config.schema import DatasetConfig
from .base import CameraIntrinsics, Dataset
from .catalog import get_kitti_base_dir

logger = get_logger(__name__)


class KittiDataset(Dataset):
    """
    Loader for KITTI stereo dataset format.

    The KITTI dataset format consists of:
    - Left color images in image_2/ directory (PNG format)
    - Right color images in image_3/ directory (PNG format, optional)
    - Sequential filenames: 000000.png, 000001.png, ...

    For fog perturbation, we only need image_2/ (left camera).
    Camera calibration, timestamps, and ground truth are handled by ORB-SLAM3.

    Supports sequence-based configuration:
        dataset:
          type: kitti
          sequence: "04"  # Resolves to ./datasets/kitti/sequences/04
    """

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        """
        Resolve KITTI dataset path from sequence number.

        KITTI datasets must be downloaded manually. This method resolves
        the sequence number to the expected path location.

        Args:
            config: Dataset configuration with either path or sequence

        Returns:
            Resolved path string

        Raises:
            ValueError: If sequence not found at expected location
        """
        # If path is provided, use it directly
        if config.path:
            return config.path

        # Must have sequence for auto-resolution
        if not config.sequence:
            raise ValueError(
                "KITTI dataset requires either 'path' or 'sequence'. "
                "Use a zero-padded odometry sequence such as '04'."
            )

        # Normalize sequence number (e.g., "4" -> "04")
        seq_num = str(config.sequence).zfill(2)
        kitti_base = get_kitti_base_dir()
        sequence_path = kitti_base / "sequences" / seq_num

        if not sequence_path.exists():
            raise ValueError(
                f"KITTI sequence '{seq_num}' not found at {sequence_path}\n\n"
                f"Please download KITTI odometry dataset from:\n"
                f"  https://www.cvlibs.net/datasets/kitti/eval_odometry.php\n\n"
                f"Extract to: {kitti_base}/\n"
                f"Expected structure:\n"
                f"  {kitti_base}/sequences/{seq_num}/image_2/\n"
                f"  {kitti_base}/sequences/{seq_num}/image_3/\n"
                f"  {kitti_base}/poses/{seq_num}.txt"
            )

        logger.info(f"Resolved KITTI sequence '{seq_num}' to: {sequence_path}")
        return str(sequence_path)

    def __init__(self, config: DatasetConfig):
        """
        Initialize KITTI dataset.

        Args:
            config: Dataset configuration
        """
        self._load_stereo = getattr(config, 'load_stereo', False)

        super().__init__(config)

    @property
    def is_stereo(self) -> bool:
        """Check if this KITTI dataset is in stereo mode.

        Returns:
            True if stereo mode is enabled (image_3 loaded), False otherwise.
        """
        return self._load_stereo

    @property
    def supports_stereo(self) -> bool:
        """KITTI dataset format supports stereo camera pairs."""
        return True

    def _normalize_sequence_for_compare(self, sequence: str) -> str:
        """Normalize KITTI sequence labels for config/path consistency checks."""
        normalized = super()._normalize_sequence_for_compare(sequence)
        if normalized.isdigit():
            return normalized.zfill(2)
        return normalized

    def _load_dataset(self) -> None:
        """Load KITTI dataset by scanning image_2/ directory."""
        logger.info(f"Loading KITTI dataset from {self.path}")

        image2_dir = self.path / 'image_2'

        if not image2_dir.exists():
            raise FileNotFoundError(
                f"KITTI image_2/ directory not found at {image2_dir}\n"
                f"Expected structure:\n"
                f"  {self.path}/\n"
                f"    └── image_2/\n"
                f"          ├── 000000.png\n"
                f"          ├── 000001.png\n"
                f"          └── ..."
            )

        image_files = sorted(image2_dir.glob('*.png'))

        if not image_files:
            raise RuntimeError(f"No PNG images found in {image2_dir}")

        image3_dir = self.path / 'image_3'
        has_stereo = self._load_stereo
        if self._load_stereo and not image3_dir.exists():
            raise FileNotFoundError(
                f"Stereo mode requested but image_3/ directory not found at {image3_dir}. "
                "Stereo mode requires both image_2/ and image_3/ directories."
            )
        if has_stereo:
            missing_right = [
                img_path.name for img_path in image_files
                if not (image3_dir / img_path.name).exists()
            ]
            if missing_right:
                sample = ", ".join(missing_right[:5])
                remaining = len(missing_right) - min(len(missing_right), 5)
                suffix = f" (and {remaining} more)" if remaining > 0 else ""
                raise FileNotFoundError(
                    f"KITTI stereo mode requires right-camera image for every left frame, "
                    f"but {len(missing_right)} files are missing in {image3_dir}: "
                    f"{sample}{suffix}"
                )

        # Extract sequence ID from path (e.g., "04" from "datasets/kitti/04")
        sequence_id = self.sequence_name

        for idx, img_path in enumerate(image_files):
            frame_info = {
                'image_path': str(img_path),
                'timestamp': float(idx),  # Use frame index as timestamp
                'sequence_id': sequence_id,
                'frame_id': idx,
                'filename': img_path.name
            }

            if has_stereo:
                image3_path = image3_dir / img_path.name
                frame_info['image_path_right'] = str(image3_path)

            self._frames.append(frame_info)

        if has_stereo:
            logger.info(f"Loaded {len(self._frames)} stereo frames from KITTI sequence {sequence_id}")
        else:
            logger.info(f"Loaded {len(self._frames)} frames from KITTI sequence {sequence_id}")

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
                f"KITTI frame {idx}: failed to load left image '{image_path}': {e}"
            ) from e

        frame_data = {
            'image': image,
            'depth': None,  # KITTI doesn't have native depth - handled by fog module
            'timestamp': frame_info['timestamp'],
            'sequence_id': frame_info['sequence_id'],
            'frame_id': frame_info['frame_id'],
            'rgb_filename': frame_info['filename']  # Original filename
        }

        if 'image_path_right' in frame_info:
            image3_path = frame_info['image_path_right']
            try:
                image3 = read_image(image3_path)
                frame_data['image_right'] = image3
                frame_data['rgb_filename_right'] = Path(image3_path).name
            except Exception as e:
                raise RuntimeError(
                    f"KITTI frame {idx}: failed to load right stereo image "
                    f"'{image3_path}': {e}. "
                    f"Stereo mode requires both left and right images."
                ) from e

        return frame_data

    def get_sequence_info(self) -> Dict[str, Any]:
        """
        Get dataset sequence information.

        Returns:
            Sequence metadata
        """
        if not self._frames:
            return {
                'sequences': [],
                'frames_per_sequence': {},
                'total_frames': 0
            }

        sequences = list(set(f['sequence_id'] for f in self._frames))

        frames_per_sequence = {}
        for seq_id in sequences:
            count = sum(1 for f in self._frames if f['sequence_id'] == seq_id)
            frames_per_sequence[seq_id] = count

        return {
            'sequences': sequences,
            'frames_per_sequence': frames_per_sequence,
            'total_frames': len(self._frames)
        }

    def get_ground_truth_path(self) -> Optional[Path]:
        """Get the path to the KITTI ground truth trajectory file.

        KITTI ground truth is stored in: datasets/kitti/poses/{sequence}.txt
        Relative to sequence directory: ../../poses/{sequence}.txt

        Returns:
            Path to ground truth file, or None if not available
        """
        # Extract sequence number from path (e.g., "04" from "datasets/kitti/sequences/04")
        sequence = self.path.name

        # KITTI poses are in the parent directory structure: ../../poses/{sequence}.txt
        gt_path = self.path.parent.parent / "poses" / f"{sequence}.txt"

        if gt_path.exists():
            return gt_path

        logger.debug(f"KITTI ground truth not found at {gt_path}")
        return None

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Get list of KITTI metadata files with destination filenames.

        Returns:
            List of (source_path, dest_name, should_truncate) tuples:
            - times.txt -> times.txt (truncate)
            - calib.txt -> calib.txt (no truncate)
        """
        metadata_files = []

        times_file = self.path / "times.txt"
        if times_file.exists():
            metadata_files.append((times_file, "times.txt", True))

        calib_file = self.path / "calib.txt"
        if calib_file.exists():
            metadata_files.append((calib_file, "calib.txt", False))

        return metadata_files

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Get the KITTI image directory name.

        Args:
            camera: "left" for left camera, "right" for right camera

        Returns:
            "image_2" for left, "image_3" for right
        """
        if camera == "right":
            return "image_3"
        return "image_2"

    def get_available_depth_dir(self, camera: str = "left") -> Optional[Path]:
        """Get reusable depth directory for a KITTI camera role.

        Priority order:
        1. FoundationStereo depth cache
        2. DA3 depth cache
        3. Legacy generic depth cache (`{camera}_depth`)

        Args:
            camera: Camera role ("left" or "right")

        Returns:
            Path to reusable depth directory, or None if no valid depth cache exists.
        """
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera role '{camera}'. Expected 'left' or 'right'."
            )

        image_dir = self.path / self.get_image_directory_name(camera)
        image_files = {p.name for p in image_dir.glob("*.png")}
        if not image_files:
            image_files = {p.name for p in image_dir.glob("*.jpg")}

        candidates = [
            self.path / f"{camera}_foundation_stereo_depth",
            self.path / f"{camera}_da3_depth",
            self.path / f"{camera}_depth",
        ]

        for depth_dir in candidates:
            if not depth_dir.exists() or not depth_dir.is_dir():
                continue

            depth_files = {p.name for p in depth_dir.glob("*.png")}
            if not depth_files:
                depth_files = {p.name for p in depth_dir.glob("*.jpg")}
            if not depth_files:
                continue

            # Require filename alignment with source images to avoid wrong-depth reuse.
            if image_files and not image_files.issubset(depth_files):
                missing = len(image_files - depth_files)
                logger.warning(
                    f"Depth directory {depth_dir} missing {missing} KITTI frame(s) for {camera}; skipping reuse"
                )
                continue

            return depth_dir

        return None

    def get_timestamps_file_path(self) -> Optional[Path]:
        """Get the path to KITTI times.txt file.

        Returns:
            Path to times.txt if it exists, None otherwise.
        """
        times_file = self.path / "times.txt"
        return times_file if times_file.exists() else None

    def get_algorithm_timestamps(self) -> Dict[int, float]:
        """Return KITTI timestamps from times.txt mapped by frame index."""
        times_file = self.path / "times.txt"
        if not times_file.exists():
            raise FileNotFoundError(
                f"KITTI times.txt is required for trajectory timestamp mapping: {times_file}"
            )

        raw_timestamps: List[float] = []
        with open(times_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    raw_timestamps.append(float(stripped))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid KITTI timestamp in {times_file} at line {line_num}: {stripped!r}"
                    ) from e

        frame_count = len(self._frames)
        if len(raw_timestamps) < frame_count:
            raise ValueError(
                f"KITTI times.txt has {len(raw_timestamps)} entries but dataset has {frame_count} frames."
            )

        return {idx: raw_timestamps[idx] for idx in range(frame_count)}

    def _extract_projection_intrinsics(
        self,
        calib_path: Path,
        matrix_key: str,
    ) -> Optional[CameraIntrinsics]:
        """Extract fx/fy/cx/cy from a KITTI projection matrix entry."""
        try:
            with open(calib_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith(f"{matrix_key}:"):
                        continue

                    values = [float(v) for v in line.split(":", 1)[1].strip().split()]
                    if len(values) != 12:
                        raise ValueError(
                            f"Invalid {matrix_key} projection matrix length in {calib_path}: "
                            f"expected 12 values, got {len(values)}"
                        )

                    return CameraIntrinsics(
                        fx=values[0],
                        fy=values[5],
                        cx=values[2],
                        cy=values[6],
                    )
        except (OSError, ValueError) as e:
            logger.error(f"Failed to parse KITTI matrix {matrix_key}: {e}")
            return None

        logger.warning(f"KITTI matrix {matrix_key} not found in {calib_path}")
        return None

    def get_camera_intrinsics(self, camera: str = "left") -> Optional[CameraIntrinsics]:
        """Get per-camera intrinsics for KITTI from calib.txt."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )

        calib_path = self.path / "calib.txt"
        if not calib_path.exists():
            logger.warning(f"KITTI calibration file not found in {self.path}")
            return None

        matrix_key = "P2" if camera == "left" else "P3"
        intrinsics = self._extract_projection_intrinsics(calib_path, matrix_key)
        if intrinsics is None:
            return None

        try:
            stereo_calib = parse_kitti_calibration(calib_path)
            baseline = float(stereo_calib.get("baseline", 0.0))
            if baseline > 0:
                intrinsics = CameraIntrinsics(
                    fx=intrinsics.fx,
                    fy=intrinsics.fy,
                    cx=intrinsics.cx,
                    cy=intrinsics.cy,
                    width=intrinsics.width,
                    height=intrinsics.height,
                    baseline=baseline,
                )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Failed to parse KITTI baseline from {calib_path}: {e}")

        return intrinsics

    def create_pyslam_structure(
        self,
        images_path: Path,
        temp_root: Path,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Create KITTI structure for PySLAM.

        PySLAM expects:
            base_path/sequences/{seq}/image_2/*.png
            base_path/sequences/{seq}/image_3/*.png
            base_path/poses/{seq}.txt

        Args:
            images_path: Path to images (perturbed output or original dataset)
            temp_root: Temporary directory root to create structure in
            max_frames: Maximum number of frames to include

        Returns:
            Path to the created dataset structure root
        """
        sequence_name = self.sequence_name
        seq_dir = temp_root / "sequences" / sequence_name
        poses_dir = temp_root / "poses"

        seq_dir.mkdir(parents=True, exist_ok=True)
        poses_dir.mkdir(parents=True, exist_ok=True)

        left_dir_name = self.get_image_directory_name("left")
        camera_roles = self.get_active_camera_roles()
        right_dir_name = self.get_image_directory_name("right") if "right" in camera_roles else None
        if (images_path / left_dir_name).exists():
            left_images_dir = images_path / left_dir_name
            right_images_dir = images_path / right_dir_name if right_dir_name else None
        else:
            # images_path is already the image directory
            left_images_dir = images_path
            right_images_dir = images_path.parent / right_dir_name if right_dir_name else None

        # Symlink images
        self._symlink_image_dir(left_images_dir, seq_dir / "image_2", max_frames)
        if "right" in camera_roles:
            if right_images_dir is None or not right_images_dir.exists():
                raise FileNotFoundError(
                    f"KITTI active stereo mode requires right camera directory at {right_images_dir}, "
                    "but it was not found while creating PySLAM structure."
                )
            self._symlink_image_dir(right_images_dir, seq_dir / "image_3", max_frames)

        for file_path, dest_name, should_truncate in self.get_metadata_files_with_dest():
            if file_path.exists():
                if should_truncate and max_frames:
                    self._truncate_text_file(
                        file_path, seq_dir / dest_name, max_frames, preserve_comments=False
                    )
                else:
                    shutil.copy(file_path, seq_dir / dest_name)
                logger.debug(f"{'Truncated' if should_truncate else 'Copied'} {dest_name}")

        gt_path = self.get_ground_truth_path()
        if gt_path and gt_path.exists():
            os.symlink(gt_path, poses_dir / f"{sequence_name}.txt")
            logger.debug(f"Symlinked poses: {gt_path}")

        logger.info(f"Created KITTI pyslam structure at {temp_root}")
        return temp_root

    def create_truncated_copy(self, max_frames: int, output_dir: Optional[Path] = None) -> Path:
        """Create a truncated copy of this KITTI dataset.

        Creates hardlinks to the first max_frames images and truncates times.txt.
        Creates both image_0/image_1 (ORB-SLAM3 format) and image_2/image_3 (standard).

        Args:
            max_frames: Maximum number of frames to include
            output_dir: Optional output directory. If None, creates a temp directory.

        Returns:
            Path to the truncated dataset directory
        """
        if output_dir is None:
            output_dir = create_temp_dir(prefix="kitti_truncated_")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        for src_name, dst_names in [
            ("image_2", ["image_0", "image_2"]),
            ("image_3", ["image_1", "image_3"])
        ]:
            src_dir = self.path / src_name
            if src_dir.exists():
                img_files = sorted(src_dir.glob("*.png"))[:max_frames]
                if not img_files:
                    img_files = sorted(src_dir.glob("*.jpg"))[:max_frames]

                for dst_name in dst_names:
                    dst_dir = output_dir / dst_name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    for img_file in img_files:
                        self._link_or_copy_file(img_file, dst_dir / img_file.name)

        # Truncate times.txt
        times_file = self.path / "times.txt"
        if times_file.exists():
            self._truncate_text_file(
                times_file,
                output_dir / "times.txt",
                max_frames,
                preserve_comments=False
            )

        calib_file = self.path / "calib.txt"
        if calib_file.exists():
            self._link_or_copy_file(calib_file, output_dir / "calib.txt")

        logger.info(f"Created truncated KITTI dataset ({max_frames} frames) at {output_dir}")
        return output_dir
