"""EuRoC MAV dataset loader."""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import yaml
import numpy as np

from ..utils.paths import create_temp_dir

from ..utils import get_logger, read_image, parse_euroc_calibration
from ..config.schema import DatasetConfig
from .base import CameraIntrinsics, Dataset
from .catalog import get_euroc_sequence, list_euroc_sequences, get_euroc_base_dir
from .download import ensure_euroc_sequence

logger = get_logger(__name__)


class EuRoCDataset(Dataset):
    """
    Loader for EuRoC MAV dataset format.

    The EuRoC dataset format consists of:
    - Left camera images in mav0/cam0/data/ (PNG format, grayscale)
    - Right camera images in mav0/cam1/data/ (PNG format, grayscale)
    - Camera timestamps in mav0/cam0/data.csv and mav0/cam1/data.csv
    - IMU data in mav0/imu0/data.csv
    - Ground truth in mav0/state_groundtruth_estimate0/data.csv

    Supports sequence-based configuration with auto-download:
        dataset:
          type: euroc
          sequence: V1_01_easy  # Auto-downloads if not present
    """

    # Timestamp is in nanoseconds, convert to seconds
    TIMESTAMP_SCALE = 1e-9

    @classmethod
    def resolve_path(cls, config: DatasetConfig) -> str:
        """
        Resolve EuRoC dataset path, downloading if necessary.

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
                "EuRoC dataset requires either 'path' or 'sequence'. "
                "Example sequences: V1_01_easy, MH_01_easy, V2_02_medium"
            )

        entry = get_euroc_sequence(config.sequence)
        if entry is None:
            available = list_euroc_sequences()
            raise ValueError(
                f"Unknown EuRoC sequence '{config.sequence}'.\n"
                f"Available sequences: {', '.join(sorted(available))}"
            )

        # ensure_euroc_sequence handles download if needed
        sequence_path = ensure_euroc_sequence(config.sequence)
        logger.info(f"Resolved EuRoC sequence '{config.sequence}' to: {sequence_path}")
        return str(sequence_path)

    def __init__(self, config: DatasetConfig):
        """
        Initialize EuRoC dataset.

        Args:
            config: Dataset configuration
        """
        self.load_stereo = getattr(config, 'load_stereo', False)

        super().__init__(config)

    def _normalize_sequence_for_compare(self, sequence: str) -> str:
        """Normalize EuRoC sequence labels for config/path consistency checks."""
        normalized = super()._normalize_sequence_for_compare(sequence)
        return normalized.lower()

    def _load_dataset(self) -> None:
        """Load EuRoC dataset by reading camera CSV files."""
        logger.info(f"Loading EuRoC dataset from {self.path}")

        cam0_data_dir = self.path / 'mav0' / 'cam0' / 'data'
        cam0_csv = self.path / 'mav0' / 'cam0' / 'data.csv'

        if not cam0_data_dir.exists():
            raise FileNotFoundError(
                f"EuRoC cam0/data/ directory not found at {cam0_data_dir}\n"
                f"Expected structure:\n"
                f"  {self.path}/\n"
                f"    └── mav0/\n"
                f"          ├── cam0/\n"
                f"          │     ├── data/\n"
                f"          │     │     ├── 1403636579763555584.png\n"
                f"          │     │     └── ...\n"
                f"          │     └── data.csv\n"
                f"          └── cam1/ (for stereo)"
            )

        cam1_data_dir = self.path / 'mav0' / 'cam1' / 'data'
        has_stereo = self.load_stereo
        if self.load_stereo and not cam1_data_dir.exists():
            raise FileNotFoundError(
                f"Stereo mode requested but cam1/data/ directory not found at {cam1_data_dir}. "
                "Stereo mode requires both mav0/cam0/data/ and mav0/cam1/data/ directories."
            )

        sequence_id = self.sequence_name

        if cam0_csv.exists():
            timestamps_and_files = self._load_timestamps_from_csv(cam0_csv)
        else:
            timestamps_and_files = self._scan_image_directory(cam0_data_dir)

        for idx, (timestamp_ns, filename) in enumerate(timestamps_and_files):
            frame_info = {
                'image_path': str(cam0_data_dir / filename),
                'timestamp': timestamp_ns * self.TIMESTAMP_SCALE,  # Convert to seconds
                'timestamp_ns': timestamp_ns,
                'sequence_id': sequence_id,
                'frame_id': idx,
                'filename': filename,
            }

            if has_stereo:
                right_path = cam1_data_dir / filename
                if not right_path.exists():
                    raise FileNotFoundError(
                        f"EuRoC stereo mode requires right-camera image '{filename}' in {cam1_data_dir}, "
                        f"but it was not found."
                    )
                frame_info['image_path_right'] = str(right_path)

            self._frames.append(frame_info)

        self._load_ground_truth()

        logger.info(f"Loaded {len(self._frames)} frames from EuRoC sequence {sequence_id}")

    def _load_timestamps_from_csv(self, csv_path: Path) -> List[tuple]:
        """
        Load timestamps and filenames from EuRoC CSV file.

        CSV format:
        #timestamp [ns],filename
        1403636579763555584,1403636579763555584.png

        Args:
            csv_path: Path to data.csv file

        Returns:
            List of (timestamp_ns, filename) tuples
        """
        timestamps_and_files = []

        with open(csv_path, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                parts = line.split(',')
                if len(parts) >= 2:
                    timestamp_ns = int(parts[0])
                    filename = parts[1]
                    timestamps_and_files.append((timestamp_ns, filename))

        return timestamps_and_files

    def _scan_image_directory(self, image_dir: Path) -> List[tuple]:
        """
        Scan image directory for PNG files when CSV is not available.

        EuRoC images are named with their timestamp: {timestamp_ns}.png

        Args:
            image_dir: Path to image directory

        Returns:
            List of (timestamp_ns, filename) tuples sorted by timestamp
        """
        timestamps_and_files = []

        for img_path in sorted(image_dir.glob('*.png')):
            filename = img_path.name
            # Extract timestamp from filename (e.g., "1403636579763555584.png")
            try:
                timestamp_ns = int(img_path.stem)
                timestamps_and_files.append((timestamp_ns, filename))
            except ValueError:
                logger.warning(f"Could not parse timestamp from filename: {filename}")

        return timestamps_and_files

    def _load_ground_truth(self) -> None:
        """Load ground truth trajectory if available."""
        gt_csv = self.path / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'

        if not gt_csv.exists():
            # Try alternative location
            gt_csv = self.path / 'mav0' / 'groundtruth' / 'data.csv'

        if not gt_csv.exists():
            logger.debug("No ground truth file found")
            return

        logger.info(f"Loading ground truth from {gt_csv}")

        try:
            # Read ground truth CSV
            # Format: #timestamp,p_RS_R_x,p_RS_R_y,p_RS_R_z,q_RS_w,q_RS_x,q_RS_y,q_RS_z,v_RS_R_x,v_RS_R_y,v_RS_R_z,...
            timestamps = []
            positions = []
            quaternions = []

            with open(gt_csv, 'r') as f:
                for line in f:
                    line = line.strip()

                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    parts = line.split(',')

                    # Need at least timestamp + position (3) + quaternion (4)
                    if len(parts) >= 8:
                        timestamp_ns = int(parts[0])
                        # Position: x, y, z
                        px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
                        # Quaternion: w, x, y, z (EuRoC format)
                        qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])

                        timestamps.append(timestamp_ns)
                        positions.append([px, py, pz])
                        quaternions.append([qx, qy, qz, qw])  # Convert to x,y,z,w order

            if timestamps:
                self._ground_truth = self._quaternions_to_matrices(positions, quaternions)
                self._gt_timestamps = timestamps
                logger.info(f"Loaded {len(self._ground_truth)} ground truth poses")

        except Exception as e:
            raise RuntimeError(f"Failed to load EuRoC ground truth from {gt_csv}: {e}") from e

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

            if len(image.shape) == 2:
                image = np.stack([image, image, image], axis=-1)

        except Exception as e:
            raise RuntimeError(
                f"EuRoC frame {idx}: failed to load left image '{image_path}': {e}"
            ) from e

        result = {
            'image': image,
            'depth': None,  # EuRoC doesn't have depth
            'timestamp': frame_info['timestamp'],
            'sequence_id': frame_info['sequence_id'],
            'frame_id': frame_info['frame_id'],
            'rgb_filename': frame_info['filename'],
        }

        if 'image_path_right' in frame_info:
            right_path = frame_info['image_path_right']
            try:
                image_right = read_image(right_path)
                if len(image_right.shape) == 2:
                    image_right = np.stack([image_right, image_right, image_right], axis=-1)
                result['image_right'] = image_right
                result['rgb_filename_right'] = frame_info['filename']
            except Exception as e:
                raise RuntimeError(
                    f"EuRoC frame {idx}: failed to load right stereo image "
                    f"'{right_path}': {e}. Stereo mode requires both left and right images."
                ) from e

        return result

    def get_ground_truth_path(self) -> Optional[Path]:
        """Get the path to the EuRoC ground truth trajectory file.

        EuRoC ground truth is stored in mav0/state_groundtruth_estimate0/data.csv

        Returns:
            Path to ground truth CSV file, or None if not available
        """
        gt_path = self.path / 'mav0' / 'state_groundtruth_estimate0' / 'data.csv'
        if gt_path.exists():
            return gt_path

        # Try alternative location
        gt_path = self.path / 'mav0' / 'groundtruth' / 'data.csv'
        if gt_path.exists():
            return gt_path

        logger.debug(f"EuRoC ground truth not found in {self.path}")
        return None

    def get_metadata_files_with_dest(self) -> List[Tuple[Path, str, bool]]:
        """Get EuRoC metadata files with unique destination filenames.

        EuRoC has multiple data.csv and sensor.yaml files in different directories.
        This method provides unique destination names to avoid file overwrites.

        Returns:
            List of (source_path, dest_filename, should_truncate) tuples
        """
        result = []

        # Camera CSV files with unique names
        cam0_csv = self.path / 'mav0' / 'cam0' / 'data.csv'
        cam1_csv = self.path / 'mav0' / 'cam1' / 'data.csv'

        if cam0_csv.exists():
            result.append((cam0_csv, 'cam0_data.csv', True))
        if cam1_csv.exists():
            result.append((cam1_csv, 'cam1_data.csv', True))

        # Ground truth - keep original name (unique)
        gt_path = self.get_ground_truth_path()
        if gt_path:
            result.append((gt_path, gt_path.name, False))

        # Sensor calibration files with unique names
        for sensor_yaml in ['cam0/sensor.yaml', 'cam1/sensor.yaml', 'imu0/sensor.yaml']:
            yaml_path = self.path / 'mav0' / sensor_yaml
            if yaml_path.exists():
                unique_name = sensor_yaml.replace('/', '_')
                result.append((yaml_path, unique_name, False))

        return result

    def filter_metadata_for_kept_frames(
        self,
        output_dir: Path,
        kept_frame_indices: List[int],
        total_frames: int
    ) -> None:
        """Filter EuRoC data.csv files to only include kept frames.

        When frames are dropped, the data.csv files need to be updated to only
        reference the frames that were actually saved. This prevents SLAM algorithms
        from looking for frames that don't exist.

        Args:
            output_dir: Directory where perturbed output was saved
                       (e.g., results/experiment/images/module_name/)
            kept_frame_indices: List of frame indices that were kept (not dropped)
            total_frames: Total number of frames before dropping

        Raises:
            RuntimeError: If validation fails (data line count mismatch or filtered count mismatch)
        """
        kept_set = set(kept_frame_indices)
        expected_kept_count = len(kept_frame_indices)
        dropped_count = total_frames - expected_kept_count

        for csv_name in ['cam0_data.csv', 'cam1_data.csv']:
            csv_path = output_dir / csv_name
            if not csv_path.exists():
                continue

            # Read all lines
            with open(csv_path, 'r') as f:
                lines = f.readlines()

            original_data_count = sum(
                1 for line in lines
                if line.strip() and not line.strip().startswith('#')
            )

            if original_data_count != total_frames:
                raise RuntimeError(
                    f"Metadata validation failed for {csv_name}: "
                    f"expected {total_frames} data lines but found {original_data_count}. "
                    f"The metadata file may have been incorrectly truncated or corrupted."
                )

            # Filter: keep comments and lines for kept frame indices
            filtered_lines = []
            data_line_idx = 0

            for line in lines:
                stripped = line.strip()
                if stripped.startswith('#') or not stripped:
                    filtered_lines.append(line)
                else:
                    # Data line - check if frame index is in kept set
                    if data_line_idx in kept_set:
                        filtered_lines.append(line)
                    data_line_idx += 1

            filtered_data_count = sum(
                1 for line in filtered_lines
                if line.strip() and not line.strip().startswith('#')
            )

            if filtered_data_count != expected_kept_count:
                raise RuntimeError(
                    f"Metadata filtering validation failed for {csv_name}: "
                    f"expected {expected_kept_count} entries after filtering but got {filtered_data_count}. "
                    f"Kept indices: {sorted(kept_frame_indices)[:10]}{'...' if len(kept_frame_indices) > 10 else ''}"
                )

            # Overwrite with filtered content
            with open(csv_path, 'w') as f:
                f.writelines(filtered_lines)

            logger.info(
                f"Filtered {csv_name}: kept {filtered_data_count}/{total_frames} entries "
                f"({dropped_count} dropped) - validated"
            )

    def get_image_directory_name(self, camera: str = "left") -> str:
        """Get the EuRoC image directory name.

        Args:
            camera: "left" for cam0, "right" for cam1

        Returns:
            Directory path relative to dataset root (e.g., "mav0/cam0/data")
        """
        if camera == "right":
            return "mav0/cam1/data"
        return "mav0/cam0/data"

    def get_depth_directory_path(self) -> Optional[Path]:
        """Get the path to depth images directory.

        EuRoC doesn't have native depth data.

        Returns:
            None (EuRoC is stereo-only, no depth maps)
        """
        return None

    def requires_association_file(self) -> bool:
        """EuRoC doesn't require an association file.

        Timestamps are embedded in image filenames and CSV files.

        Returns:
            False - EuRoC uses timestamp-named files
        """
        return False

    def get_timestamps_file_path(self) -> Optional[Path]:
        """Get the path to the timestamps file.

        EuRoC stores timestamps in cam0/data.csv

        Returns:
            Path to cam0/data.csv if it exists
        """
        csv_path = self.path / 'mav0' / 'cam0' / 'data.csv'
        return csv_path if csv_path.exists() else None

    def get_algorithm_timestamps(self) -> Dict[int, int]:
        """Return EuRoC timestamps in native nanoseconds for algorithm conversion."""
        timestamps_ns: Dict[int, int] = {}
        for idx, frame_info in enumerate(self._frames):
            ts_ns = frame_info.get("timestamp_ns")
            if ts_ns is None:
                raise ValueError(
                    f"EuRoC frame {idx} is missing 'timestamp_ns' required for evaluation."
                )
            if not isinstance(ts_ns, (int, np.integer)):
                raise ValueError(
                    f"EuRoC frame {idx} has non-integer timestamp_ns: {ts_ns!r}"
                )
            timestamps_ns[idx] = int(ts_ns)
        return timestamps_ns

    def _extract_sensor_intrinsics(self, sensor_yaml: Path) -> Optional[CameraIntrinsics]:
        """Extract fx/fy/cx/cy (and resolution when present) from a sensor.yaml file."""
        try:
            with open(sensor_yaml, "r", encoding="utf-8") as f:
                sensor_data = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"Failed to read EuRoC calibration file {sensor_yaml}: {e}")
            return None

        intrinsics = sensor_data.get("intrinsics")
        if intrinsics is None:
            camera_data = sensor_data.get("camera", {})
            intrinsics = camera_data.get("intrinsics")

        if not isinstance(intrinsics, (list, tuple)) or len(intrinsics) < 4:
            logger.error(
                f"Invalid intrinsics in {sensor_yaml}. Expected [fx, fy, cx, cy], got: {intrinsics}"
            )
            return None

        fx = float(intrinsics[0])
        fy = float(intrinsics[1])
        cx = float(intrinsics[2])
        cy = float(intrinsics[3])

        resolution = sensor_data.get("resolution")
        width = height = None
        if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
            width, height = resolution[0], resolution[1]
        else:
            camera_data = sensor_data.get("camera", {})
            width = camera_data.get("image_width")
            height = camera_data.get("image_height")

        width_value = float(width) if width is not None else None
        height_value = float(height) if height is not None else None

        return CameraIntrinsics(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            width=width_value,
            height=height_value,
        )

    def get_camera_intrinsics(self, camera: str = "left") -> Optional[CameraIntrinsics]:
        """Get per-camera intrinsics for EuRoC from sensor.yaml files."""
        if camera not in {"left", "right"}:
            raise ValueError(
                f"Unsupported camera '{camera}'. Expected 'left' or 'right'."
            )

        cam_dir = "cam0" if camera == "left" else "cam1"
        sensor_yaml = self.path / "mav0" / cam_dir / "sensor.yaml"
        if not sensor_yaml.exists():
            logger.warning(f"EuRoC {cam_dir} sensor.yaml not found: {sensor_yaml}")
            return None

        intrinsics = self._extract_sensor_intrinsics(sensor_yaml)
        if intrinsics is None:
            return None

        cam0_yaml = self.path / "mav0" / "cam0" / "sensor.yaml"
        cam1_yaml = self.path / "mav0" / "cam1" / "sensor.yaml"
        if cam0_yaml.exists() and cam1_yaml.exists():
            try:
                stereo_calib = parse_euroc_calibration(cam0_yaml, cam1_yaml)
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
                logger.warning(f"Failed to parse EuRoC baseline from sensor.yaml files: {e}")

        return intrinsics

    @property
    def is_stereo(self) -> bool:
        """Check if this EuRoC dataset is in stereo mode.

        Returns:
            True if stereo mode is enabled (cam1 loaded), False otherwise.
        """
        return self.load_stereo

    @property
    def supports_stereo(self) -> bool:
        """EuRoC dataset format supports stereo camera pairs."""
        return True

    def create_pyslam_structure(
        self,
        images_path: Path,
        temp_root: Path,
        max_frames: Optional[int] = None,
    ) -> Path:
        """Create EuRoC structure for PySLAM.

        PySLAM expects:
            base_path/{seq_name}/mav0/cam0/data/*.png
            base_path/{seq_name}/mav0/cam1/data/*.png
            base_path/{seq_name}/mav0/cam0/data.csv
            base_path/{seq_name}/mav0/cam1/data.csv

        Args:
            images_path: Path to images (perturbed output or original dataset)
            temp_root: Temporary directory root to create structure in
            max_frames: Maximum number of frames to include

        Returns:
            Path to the created dataset structure root
        """
        sequence_name = self.sequence_name
        seq_dir = temp_root / sequence_name
        mav0_dir = seq_dir / "mav0"

        left_dir_name = self.get_image_directory_name("left")   # mav0/cam0/data
        camera_roles = self.get_active_camera_roles()
        right_dir_name = self.get_image_directory_name("right") if "right" in camera_roles else None

        if (images_path / left_dir_name).exists():
            left_images_dir = images_path / left_dir_name
            right_images_dir = images_path / right_dir_name if right_dir_name else None
        elif (images_path / "image_2").exists():
            left_images_dir = images_path / "image_2"
            right_images_dir = images_path / "image_3" if right_dir_name else None
        else:
            left_images_dir = images_path
            right_images_dir = images_path.parent / "cam1" / "data" if right_dir_name else None

        cam0_data_dir = mav0_dir / "cam0" / "data"
        cam1_data_dir = mav0_dir / "cam1" / "data"

        self._symlink_image_dir(left_images_dir, cam0_data_dir, max_frames)
        if "right" in camera_roles:
            if right_images_dir is None or not right_images_dir.exists():
                raise FileNotFoundError(
                    f"EuRoC active stereo mode requires right camera directory at {right_images_dir}, "
                    "but it was not found while creating PySLAM structure."
                )
            self._symlink_image_dir(right_images_dir, cam1_data_dir, max_frames)

        for file_path, _dest_name, should_truncate in self.get_metadata_files_with_dest():
            if file_path.exists():
                # Preserve EuRoC directory structure for metadata
                # e.g., mav0/cam0/data.csv -> mav0/cam0/data.csv
                rel_path = file_path.relative_to(self.path)
                dest_path = seq_dir / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)

                if should_truncate and max_frames:
                    self._truncate_text_file(
                        file_path, dest_path, max_frames, preserve_comments=True
                    )
                else:
                    shutil.copy(file_path, dest_path)
                logger.debug(f"{'Truncated' if should_truncate else 'Copied'} {rel_path}")

        logger.info(f"Created EuRoC pyslam structure at {temp_root}")
        return temp_root

    def create_truncated_copy(self, max_frames: int, output_dir: Optional[Path] = None) -> Path:
        """Create a truncated copy of this EuRoC dataset.

        Creates hardlinks to the first max_frames images and truncates metadata files.

        Args:
            max_frames: Maximum number of frames to include
            output_dir: Optional output directory. If None, creates a temp directory.

        Returns:
            Path to the truncated dataset directory
        """
        if output_dir is None:
            output_dir = create_temp_dir(prefix="euroc_truncated_")
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        mav0_dir = output_dir / 'mav0'

        for cam_dir in ['cam0', 'cam1']:
            src_data_dir = self.path / 'mav0' / cam_dir / 'data'
            if src_data_dir.exists():
                dst_data_dir = mav0_dir / cam_dir / 'data'
                dst_data_dir.mkdir(parents=True, exist_ok=True)

                img_files = sorted(src_data_dir.glob("*.png"))[:max_frames]

                for img_file in img_files:
                    self._link_or_copy_file(img_file, dst_data_dir / img_file.name)

                # Truncate data.csv
                src_csv = self.path / 'mav0' / cam_dir / 'data.csv'
                if src_csv.exists():
                    dst_csv = mav0_dir / cam_dir / 'data.csv'
                    self._truncate_text_file(
                        src_csv,
                        dst_csv,
                        max_frames,
                        preserve_comments=True
                    )

                src_yaml = self.path / 'mav0' / cam_dir / 'sensor.yaml'
                if src_yaml.exists():
                    dst_yaml = mav0_dir / cam_dir / 'sensor.yaml'
                    self._link_or_copy_file(src_yaml, dst_yaml)

        src_imu_dir = self.path / 'mav0' / 'imu0'
        if src_imu_dir.exists():
            dst_imu_dir = mav0_dir / 'imu0'
            dst_imu_dir.mkdir(parents=True, exist_ok=True)

            src_imu_csv = src_imu_dir / 'data.csv'
            if src_imu_csv.exists():
                self._link_or_copy_file(src_imu_csv, dst_imu_dir / 'data.csv')

            src_imu_yaml = src_imu_dir / 'sensor.yaml'
            if src_imu_yaml.exists():
                self._link_or_copy_file(src_imu_yaml, dst_imu_dir / 'sensor.yaml')

        gt_path = self.get_ground_truth_path()
        if gt_path:
            rel_path = gt_path.relative_to(self.path)
            dst_gt = output_dir / rel_path
            dst_gt.parent.mkdir(parents=True, exist_ok=True)
            self._link_or_copy_file(gt_path, dst_gt)

        # Generate ORB-SLAM3 compatible timestamps file
        # ORB-SLAM3 expects a simple file with one timestamp (nanoseconds) per line
        orbslam_timestamps_path = output_dir / "orbslam3_timestamps.txt"
        self._generate_orbslam_timestamps(mav0_dir / 'cam0' / 'data.csv', orbslam_timestamps_path)

        logger.info(f"Created truncated EuRoC dataset ({max_frames} frames) at {output_dir}")
        return output_dir

    def _generate_orbslam_timestamps(self, data_csv_path: Path, output_path: Path) -> None:
        """Generate ORB-SLAM3 compatible timestamps file from truncated data.csv.

        ORB-SLAM3 expects a simple text file with one timestamp per line in nanoseconds.

        Args:
            data_csv_path: Path to the truncated data.csv
            output_path: Path to write the timestamps file
        """
        if not data_csv_path.exists():
            logger.warning(f"Cannot generate ORB-SLAM3 timestamps: {data_csv_path} not found")
            return

        timestamps = []
        with open(data_csv_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Extract timestamp (first column before comma)
                parts = line.split(',')
                if parts:
                    timestamps.append(parts[0])

        with open(output_path, 'w') as f:
            for ts in timestamps:
                f.write(f"{ts}\n")

        logger.debug(f"Generated ORB-SLAM3 timestamps file with {len(timestamps)} entries: {output_path}")
