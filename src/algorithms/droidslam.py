"""DROID-SLAM algorithm implementation.

DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras.
Uses a conda environment with PyTorch.
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


# TUM freiburg calibration parameters (fx, fy, cx, cy, k1, k2, p1, p2, k3)
TUM_CALIBRATIONS = {
    "freiburg1": "517.3 516.5 318.6 255.3 0.2624 -0.9531 -0.0054 0.0026 1.1633",
    "freiburg2": "520.9 521.0 325.1 249.7 0.2312 -0.7849 -0.0033 -0.0001 0.9172",
    "freiburg3": "535.4 539.2 320.1 247.6",
}


class DROIDSLAMAlgorithm(SLAMAlgorithm):
    """DROID-SLAM via conda environment. Deep visual SLAM supporting mono, stereo, and RGB-D."""

    def __init__(self):
        self.droid_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "DROID-SLAM"
        self.conda_env = "droidslam"
        self._process = None

    @property
    def name(self) -> str:
        return "droidslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "tum": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        """DROID-SLAM doesn't use config files - it uses calibration files.

        For TUM, we detect the freiburg version from sequence name.
        Returns the calibration file path or identifier.
        """
        if dataset_type.lower() == "tum":
            seq_lower = str(sequence).lower()
            if "freiburg1" in seq_lower or "fr1" in seq_lower:
                return "tum1"
            elif "freiburg2" in seq_lower or "fr2" in seq_lower:
                return "tum2"
            elif "freiburg3" in seq_lower or "fr3" in seq_lower:
                return "tum3"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """DROID-SLAM uses calibration identifiers, not internal config file paths."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate DROID-SLAM runtime dependencies."""
        if not self.droid_path.exists():
            raise RuntimeError(f"DROID-SLAM not found at {self.droid_path}")

        weights_path = self.droid_path / "droid.pth"
        if not weights_path.exists():
            raise RuntimeError(
                f"DROID-SLAM weights not found at {weights_path}. "
                "Run: cd deps/slam-algorithms/DROID-SLAM && ./tools/download_model.sh"
            )

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by DROID-SLAM execution."""
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)
        return {
            "dataset_path": request.dataset_path,
            "slam_config": ctx.internal_config_name or request.slam_config,
            "output_dir": request.output_dir,
            "dataset_type": request.dataset_type,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "camera_paths": request.extras.get("camera_paths", {}),
            "timestamps_by_frame": timestamps_by_frame,
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        dataset_path = inputs["dataset_path"]
        slam_config = inputs["slam_config"]
        output_dir = inputs["output_dir"]
        dataset_type = inputs["dataset_type"]
        camera_paths = inputs["camera_paths"]
        timestamps_by_frame = inputs["timestamps_by_frame"]

        # Prepare calibration file
        calib_file = self._prepare_calibration(dataset_path, slam_config, dataset_type)
        if not calib_file:
            return None

        # Prepare dataset path (find rgb directory)
        image_dir = self._prepare_dataset(dataset_path, dataset_type, camera_paths)
        if not image_dir:
            return None

        return ExecutionSpec(
            cmd=["droidslam"],
            custom_runner=lambda _spec, image_dir=image_dir, calib_file=calib_file, output_dir=output_dir, dataset_type=dataset_type, timestamps_by_frame=timestamps_by_frame: self._run_droidslam(
                image_dir,
                calib_file,
                output_dir,
                dataset_type,
                timestamps_by_frame,
            ),
            log_prefix="DROID-SLAM",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  DROID-SLAM stereo mode not yet implemented. Using mono mode.")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run DROID-SLAM: {e}")
            return False

    def _prepare_calibration(self, dataset_path: Path, slam_config: str, dataset_type: str) -> Optional[Path]:
        """Create a calibration file for DROID-SLAM."""
        logger.info("  Preparing calibration file...")

        if dataset_type == "tum":
            calib_params = None

            config_lower = slam_config.lower()
            if "tum1" in config_lower or "freiburg1" in config_lower:
                calib_params = TUM_CALIBRATIONS["freiburg1"]
            elif "tum2" in config_lower or "freiburg2" in config_lower:
                calib_params = TUM_CALIBRATIONS["freiburg2"]
            elif "tum3" in config_lower or "freiburg3" in config_lower:
                calib_params = TUM_CALIBRATIONS["freiburg3"]
            else:
                # Try to detect from path
                path_str = str(dataset_path).lower()
                if "freiburg1" in path_str or "fr1" in path_str:
                    calib_params = TUM_CALIBRATIONS["freiburg1"]
                elif "freiburg2" in path_str or "fr2" in path_str:
                    calib_params = TUM_CALIBRATIONS["freiburg2"]
                elif "freiburg3" in path_str or "fr3" in path_str:
                    calib_params = TUM_CALIBRATIONS["freiburg3"]

            if not calib_params:
                logger.error("  Could not determine TUM freiburg version for calibration")
                logger.error("  Use --slam-config tum1/tum2/tum3 to specify")
                return None

            # Write calibration file
            calib_file = dataset_path / "calib_droid.txt"
            with open(calib_file, 'w') as f:
                f.write(calib_params + "\n")
            logger.info(f"    Created calibration file: {calib_file}")
            return calib_file

        else:
            logger.error(f"  Unsupported dataset type for calibration: {dataset_type}")
            return None

    def _prepare_dataset(
        self,
        dataset_path: Path,
        dataset_type: str,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Resolve DROID-SLAM image directory from dataset runtime contract."""
        logger.info("  Preparing dataset...")

        if dataset_type != "tum":
            logger.error(f"  Unsupported dataset type: {dataset_type}")
            return None

        left_path_value = camera_paths.get("left")
        if not left_path_value:
            logger.error(
                "  Missing left camera path in request extras. "
                "Evaluation pipeline must pass dataset-resolved camera paths."
            )
            return None

        image_dir = Path(left_path_value)
        if not image_dir.exists() or not image_dir.is_dir():
            logger.error(f"  Left camera directory not found: {image_dir}")
            return None

        images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not images:
            logger.error(f"  No PNG/JPG images found in {image_dir}")
            return None

        logger.info(f"    Found {len(images)} images in {image_dir}/")
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
            timestamps[frame_idx] = float(timestamp)

        return timestamps

    def _run_droidslam(
        self,
        image_dir: Path,
        calib_file: Path,
        output_dir: Path,
        dataset_type: str,
        timestamps_by_frame: Dict[int, float],
    ) -> bool:
        """Execute DROID-SLAM."""
        logger.info("  Executing DROID-SLAM...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        # TUM evaluation uses stride=2 (every other frame)
        stride = 2 if dataset_type == "tum" else 3

        # We'll save the reconstruction to extract poses
        reconstruction_path = output_dir / "reconstruction.pth"

        python_cmd = (
            f"python demo.py "
            f"--imagedir={image_dir} "
            f"--calib={calib_file} "
            f"--weights=droid.pth "
            f"--stride={stride} "
            f"--disable_vis "
            f"--reconstruction_path={reconstruction_path}"
        )

        cmd = (
            f"cd {self.droid_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"{python_cmd}"
        )

        logger.info(f"    Running in conda env: {self.conda_env}")

        try:
            self._process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
                start_new_session=True
            )

            self._stream_process_output(self._process, "DROID-SLAM")

            self._wait_for_process(self._process, timeout_seconds=7200)

            if self._process.returncode != 0:
                logger.error(f"DROID-SLAM failed with return code {self._process.returncode}")
                return False

            logger.info("  DROID-SLAM completed successfully")

            if reconstruction_path.exists():
                return self._convert_reconstruction(
                    reconstruction_path,
                    output_dir,
                    dataset_type,
                    stride,
                    timestamps_by_frame,
                )
            else:
                logger.error("  No reconstruction file generated")
                return False

        except subprocess.TimeoutExpired:
            self._kill_process_group(self._process)
            logger.error("DROID-SLAM timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run DROID-SLAM: {e}")
            return False

    def _convert_reconstruction(
        self,
        reconstruction_path: Path,
        output_dir: Path,
        dataset_type: str,
        stride: int,
        timestamps_by_frame: Dict[int, float],
    ) -> bool:
        """Convert DROID-SLAM reconstruction.pth to TUM trajectory format."""
        logger.info("  Converting reconstruction to TUM format...")

        try:
            import torch

            data = torch.load(str(reconstruction_path), map_location='cpu')

            tstamps = data["tstamps"].numpy()
            poses = data["poses"].numpy()  # [N, 7] - tx, ty, tz, qx, qy, qz, qw (lie algebra)

            if dataset_type != "tum":
                raise ValueError(f"Unsupported dataset type for timestamp conversion: {dataset_type}")

            frame_count = len(timestamps_by_frame)
            sampled_timestamps = [
                timestamps_by_frame[idx]
                for idx in range(frame_count)
                if idx % stride == 0
            ]
            if not sampled_timestamps:
                raise ValueError(
                    f"No sampled timestamps available for stride={stride} with {frame_count} frames."
                )

            # DROID-SLAM tstamps are indices into the sampled image stream.
            real_timestamps: List[float] = []
            for raw_idx in tstamps:
                sampled_idx = int(raw_idx)
                if sampled_idx < 0 or sampled_idx >= len(sampled_timestamps):
                    raise ValueError(
                        f"DROID-SLAM timestamp index {sampled_idx} out of range for sampled stream "
                        f"of length {len(sampled_timestamps)} (stride={stride})."
                    )
                real_timestamps.append(sampled_timestamps[sampled_idx])
            tstamps = np.array(real_timestamps, dtype=np.float64)

            # DROID-SLAM outputs: tx, ty, tz, qx, qy, qz, qw
            tum_lines = []
            for i in range(len(tstamps)):
                ts = tstamps[i]
                pose = poses[i]  # [7] - position (3) + quaternion (4)
                tx, ty, tz = pose[:3]
                qx, qy, qz, qw = pose[3:]
                tum_lines.append(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")

            # Write trajectory
            traj_path = output_dir / "CameraTrajectory.txt"
            with open(traj_path, 'w') as f:
                f.write('\n'.join(tum_lines))

            logger.info(f"    Converted {len(tum_lines)} poses to TUM format")
            return True

        except Exception as e:
            logger.error(f"  Failed to convert reconstruction: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _find_trajectory(self, output_dir: Path) -> Optional[Path]:
        """Find trajectory file in output directory."""
        camera_traj = output_dir / "CameraTrajectory.txt"
        if camera_traj.exists():
            return camera_traj
        return None

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw trajectory output from DROID-SLAM."""
        return self._find_trajectory(request.output_dir)

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """DROID-SLAM conversion is handled during execution."""
        return raw_trajectory

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._process is not None:
            self._kill_process_group(self._process)
            self._process = None
