"""VGGT-SLAM algorithm implementation."""

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


class VGGTSLAMAlgorithm(SLAMAlgorithm):
    """VGGT-SLAM: Dense RGB SLAM using VGGT model.

    VGGT-SLAM is a dense RGB-only SLAM system from MIT SPARK lab that uses
    the VGGT (Visual Geometry Grounded Transformer) model from Facebook.
    It supports monocular operation and is designed for accurate camera tracking.

    Supports: EuRoC (monocular)

    Reference: https://github.com/MIT-SPARK/VGGT-SLAM
    """

    def __init__(self):
        self.vggtslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "VGGT-SLAM"
        self.conda_env = "vggt-slam"

    @property
    def name(self) -> str:
        return "vggtslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "euroc": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        """VGGT-SLAM doesn't use config files, sequence name is used directly."""
        if dataset_type.lower() == "euroc":
            return sequence  # e.g., "V1_01_easy"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """VGGT-SLAM does not use internal config file paths."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate VGGT-SLAM runtime dependencies."""
        if not self.vggtslam_path.exists():
            raise RuntimeError(f"VGGT-SLAM not found at {self.vggtslam_path}")

        if not self._check_conda_available():
            raise RuntimeError(
                f"Conda environment '{self.conda_env}' not found. "
                f"Run install script in {self.vggtslam_path}."
            )

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare EuRoC dataset image folder expected by VGGT-SLAM."""
        return self._prepare_euroc_dataset(
            request.dataset_path,
            request.extras.get("camera_paths", {}),
        )

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """VGGT-SLAM uses dataset-resolved camera paths; no staging cleanup is needed."""
        return None

    def _check_conda_available(self) -> bool:
        """Check if the conda environment exists."""
        try:
            result = subprocess.run(
                ["bash", "-c", f"source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null; conda env list | grep -q '^{self.conda_env} '"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to check conda environment '{self.conda_env}': {e}")
            return False

    def _prepare_euroc_dataset(
        self,
        dataset_path: Path,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Prepare EuRoC dataset for VGGT-SLAM.

        VGGT-SLAM expects images in a folder. We use the dataset-resolved
        left camera path from runtime request extras to avoid local path probing.

        Args:
            dataset_path: Path to EuRoC sequence or perturbed output directory
            camera_paths: Mapping of camera role -> absolute camera directory path

        Returns:
            Path to image folder for VGGT-SLAM, or None on error
        """
        left_path_value = camera_paths.get("left")
        if not left_path_value:
            logger.error(
                "  Missing left camera path in request extras. "
                "Evaluation pipeline must pass dataset-resolved camera paths."
            )
            return None

        image_dir = Path(str(left_path_value))
        if not image_dir.exists() or not image_dir.is_dir():
            logger.error(f"  Left camera directory not found: {image_dir}")
            return None

        image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not image_files:
            logger.error(f"  No PNG/JPG images found in {image_dir}")
            return None

        logger.info(f"  Using dataset-resolved left camera images at {image_dir}")
        return image_dir

    def _resolve_timestamps_by_frame(self, request: SLAMRunRequest) -> Dict[int, float]:
        """Resolve frame-indexed timestamps from runtime request extras."""
        raw = request.extras.get("timestamps_by_frame")
        if not isinstance(raw, dict) or not raw:
            raise ValueError(
                "Missing timestamps_by_frame in request extras. "
                "Evaluation pipeline must provide dataset-resolved timestamps."
            )

        timestamps: Dict[int, float] = {}
        for frame_idx, timestamp in raw.items():
            if not isinstance(frame_idx, int):
                raise ValueError(f"Invalid timestamps_by_frame key: {frame_idx!r}")
            if not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"Invalid timestamp value for frame {frame_idx}: {timestamp!r}"
                )
            timestamps[frame_idx] = timestamp
        return timestamps

    def _clear_gpu_memory(self) -> None:
        """Clear GPU memory before running SLAM."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("  Cleared GPU memory cache")
        except ImportError:
            pass

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by VGGT-SLAM execution."""
        image_folder = ctx.effective_dataset_path
        if not image_folder:
            return None
        output_dir = request.output_dir
        return {
            "image_folder": image_folder,
            "output_dir": output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "output_poses": output_dir / "poses_raw.txt",
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        image_folder = inputs["image_folder"]
        output_poses = inputs["output_poses"]
        return ExecutionSpec(
            cmd=["vggtslam"],
            custom_runner=lambda _spec, image_folder=image_folder, output_poses=output_poses: self._run_vggtslam(
                image_folder,
                output_poses,
            ),
            log_prefix="VGGT-SLAM",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        """Run VGGT-SLAM on EuRoC dataset.

        Args:
            request: Structured run request

        Returns:
            True if execution completed, False otherwise.
        """
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  VGGT-SLAM only supports monocular mode. Stereo flag ignored.")

        # Clear GPU memory before running
        self._clear_gpu_memory()

        # Run VGGT-SLAM
        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run VGGT-SLAM: {e}")
            return False

    def _run_vggtslam(self, image_folder: Path, output_path: Path) -> bool:
        """Execute VGGT-SLAM via conda environment.

        Args:
            image_folder: Path to dataset-resolved left image folder
            output_path: Path for output poses file

        Returns:
            True if successful, False otherwise
        """
        logger.info("  Executing VGGT-SLAM...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(Path.home() / "miniconda3").exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        cmd = (
            f"cd {self.vggtslam_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"python main.py "
            f"--image_folder {image_folder} "
            f"--max_loops 1 "
            f"--conf_threshold 25 "
            f"--min_disparity 50 "
            f"--submap_size 16 "
            f"--log_results "
            f"--log_path {output_path} "
            f"--skip_dense_log"
        )

        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
                start_new_session=True
            )

            self._stream_process_output(process, "VGGT-SLAM")

            self._wait_for_process(process, timeout_seconds=7200)  # 2 hour timeout

            if process.returncode != 0:
                logger.error(f"VGGT-SLAM failed with return code {process.returncode}")
                return False

            logger.info("  VGGT-SLAM completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            logger.error("VGGT-SLAM timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run VGGT-SLAM: {e}")
            return False

    def _convert_to_tum_format(
        self,
        raw_poses_path: Path,
        output_dir: Path,
        timestamps_by_frame: Dict[int, float],
    ) -> Optional[Path]:
        """Convert VGGT-SLAM output to TUM format with dataset timestamps.

        VGGT-SLAM outputs: frame_id x y z qx qy qz qw
        TUM format: timestamp x y z qx qy qz qw

        Args:
            raw_poses_path: Path to VGGT-SLAM output
            output_dir: Output directory
            timestamps_by_frame: Frame-indexed timestamp mapping from run request

        Returns:
            Path to converted trajectory file
        """
        if not raw_poses_path.exists():
            logger.error(f"Raw poses file not found: {raw_poses_path}")
            return None

        logger.info("  Converting trajectory to TUM format with dataset timestamps...")

        try:
            timestamp_values = set()
            for ts in timestamps_by_frame.values():
                ts_float = float(ts)
                if ts_float.is_integer():
                    timestamp_values.add(int(ts_float))

            tum_lines = []
            with open(raw_poses_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    parts = line.split()
                    if len(parts) != 8:
                        continue

                    frame_id = int(float(parts[0]))
                    x, y, z = parts[1], parts[2], parts[3]
                    qx, qy, qz, qw = parts[4], parts[5], parts[6], parts[7]

                    if frame_id not in timestamps_by_frame:
                        # VGGT-SLAM can emit either frame indices or dataset-native
                        # timestamp IDs as the first column. Accept exact timestamp IDs.
                        if frame_id in timestamp_values:
                            timestamp = frame_id
                        else:
                            raise ValueError(
                                f"Missing timestamp for frame index {frame_id} in timestamps_by_frame."
                            )
                    else:
                        timestamp = timestamps_by_frame[frame_id]

                    tum_lines.append(f"{timestamp} {x} {y} {z} {qx} {qy} {qz} {qw}")

            tum_path = output_dir / "CameraTrajectory.txt"
            with open(tum_path, 'w') as f:
                f.write('\n'.join(tum_lines))

            logger.info(f"    Converted {len(tum_lines)} poses to TUM format")
            return tum_path

        except Exception as e:
            raise RuntimeError(f"Failed to convert VGGT-SLAM trajectory to TUM format: {e}") from e

    def cleanup(self) -> None:
        """VGGT-SLAM does not create temporary staging artifacts."""
        return None

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw VGGT-SLAM trajectory output."""
        return request.output_dir / "poses_raw.txt"

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert VGGT-SLAM raw poses to TUM format."""
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)
        return self._convert_to_tum_format(
            raw_trajectory,
            request.output_dir,
            timestamps_by_frame,
        )
