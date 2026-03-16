"""GigaSLAM (Large-Scale Monocular SLAM with Hierarchical Gaussian Splats) algorithm implementation."""

import logging
import os
import subprocess
import shutil
import yaml
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


class GigaSLAMAlgorithm(SLAMAlgorithm):
    """GigaSLAM algorithm via conda environment. Monocular-only, KITTI dataset."""

    def __init__(self):
        self.gigaslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "GigaSLAM"
        self.conda_env = "gigaslam"
        self._temp_config: Optional[Path] = None

    @property
    def name(self) -> str:
        return "gigaslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "kitti": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        if dataset_type.lower() == "kitti":
            seq_num = int(sequence)
            # Map to available configs based on KITTI calibration groups
            if seq_num in [0, 1, 2]:
                return "kitti_00"
            elif seq_num in [4, 5, 6, 7, 8, 9, 10]:
                return "kitti_06"
            else:
                raise ValueError(f"No GigaSLAM config available for KITTI sequence {seq_num}")
        raise ValueError(f"GigaSLAM does not support dataset type: {dataset_type}")

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return self.gigaslam_path / "configs" / f"{ctx.internal_config_name}.yaml"

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate GigaSLAM runtime dependencies."""
        if not self.gigaslam_path.exists():
            raise RuntimeError(f"GigaSLAM not found at {self.gigaslam_path}")

        config_path = self._resolve_config_path(ctx)

        if config_path is None:
            raise RuntimeError("GigaSLAM external config path is not resolved in runtime context")

        if not config_path.exists():
            raise RuntimeError(f"GigaSLAM config file not found: {config_path}")

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by GigaSLAM execution."""
        dataset_path = request.dataset_path
        camera_paths = request.extras.get("camera_paths", {})
        image_dir = self._resolve_left_camera_path(camera_paths)
        if not image_dir:
            return None
        return {
            "dataset_path": dataset_path,
            "output_dir": request.output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "image_dir": image_dir,
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        dataset_path = inputs["dataset_path"]
        output_dir = inputs["output_dir"]
        image_dir = inputs["image_dir"]

        config_file = self._create_config(dataset_path, image_dir, output_dir, ctx)
        if not config_file:
            return None

        return ExecutionSpec(
            cmd=["gigaslam"],
            custom_runner=lambda _spec, config_file=config_file: self._run_gigaslam(config_file),
            log_prefix="GigaSLAM",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  GigaSLAM only supports monocular mode. Stereo flag ignored.")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run GigaSLAM: {e}")
            return False

    def _resolve_left_camera_path(self, camera_paths: Dict[str, object]) -> Optional[Path]:
        """Resolve left camera path from runtime request extras."""
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

    def _create_config(
        self,
        dataset_path: Path,
        image_dir: Path,
        output_dir: Path,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Create a config file for GigaSLAM by modifying native config."""
        config_path = self._resolve_config_path(ctx)

        if config_path is None:
            logger.error("External config path is not resolved in runtime context")
            return None

        if not config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            logger.error(f"  Available configs: {list((self.gigaslam_path / 'configs').glob('kitti_*.yaml'))}")
            return None

        logger.info(f"  Using config: {config_path}")

        try:
            with open(config_path, 'r') as f:
                config_text = f.read()

            import re
            config_text = re.sub(
                r'color_path:.*',
                f'color_path: "{image_dir}"',
                config_text
            )

            if 'Results:' not in config_text:
                config_text = re.sub(
                    r'(inherit_from:.*\n)',
                    f'\\1\nResults:\n  save_dir: "{output_dir}"\n',
                    config_text
                )
            else:
                config_text = re.sub(
                    r'save_dir:.*',
                    f'save_dir: "{output_dir}"',
                    config_text
                )

            # Write modified config
            temp_config = output_dir / "gigaslam_config.yaml"
            with open(temp_config, 'w') as f:
                f.write(config_text)

            self._temp_config = temp_config
            logger.info(f"  Created config: {temp_config}")
            return temp_config

        except Exception as e:
            raise RuntimeError(f"Failed to create GigaSLAM config: {e}") from e

    def _run_gigaslam(self, config_file: Path) -> bool:
        """Execute GigaSLAM."""
        logger.info("  Executing GigaSLAM...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        # --eval flag is required: it enables eval_rendering which saves poses_est.txt
        # Side effect: also saves per-frame PNG images (can't be disabled without modifying GigaSLAM)
        cmd = (
            f"cd {self.gigaslam_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"PYTHONUNBUFFERED=1 python -u slam.py --config {config_file} --eval"
        )

        process = None
        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
                start_new_session=True
            )
            self._current_process = process  # Store for cleanup

            self._stream_process_output(process, "GigaSLAM")

            self._wait_for_process(process, timeout_seconds=14400)  # 4 hour timeout for large-scale SLAM

            if process.returncode != 0:
                logger.error(f"GigaSLAM failed with return code {process.returncode}")
                return False

            logger.info("  GigaSLAM completed successfully")
            return True

        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            logger.error("GigaSLAM timed out after 4 hours")
            return False
        except Exception as e:
            if process:
                self._kill_process_group(process)
            logger.error(f"Failed to run GigaSLAM: {e}")
            return False
        finally:
            self._current_process = None

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw GigaSLAM trajectory output."""
        output_dir = request.output_dir

        logger.info(f"  Looking for trajectory output in: {output_dir}")

        result_dirs = list(output_dir.glob("**/poses_est.txt"))
        if not result_dirs:
            logger.error(f"Could not find poses_est.txt in {output_dir}")
            return None

        poses_file = result_dirs[0]
        logger.info(f"    Found poses at: {poses_file}")
        return poses_file

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert GigaSLAM trajectory to TUM format."""
        output_dir = request.output_dir
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)

        poses_idx_file = raw_trajectory.parent / "poses_idx.txt"

        try:
            poses = []
            with open(raw_trajectory, 'r') as f:
                for line in f:
                    values = list(map(float, line.strip().split()))
                    if len(values) == 16:
                        pose = np.array(values).reshape(4, 4)
                        poses.append(pose)
                    elif len(values) == 12:
                        pose = np.eye(4)
                        pose[:3, :] = np.array(values).reshape(3, 4)
                        poses.append(pose)

            if not poses:
                logger.error("No valid poses found in poses_est.txt")
                return None

            logger.info(f"    Loaded {len(poses)} poses")

            frame_indices: List[int] = []
            if poses_idx_file.exists():
                with open(poses_idx_file, 'r') as f:
                    frame_indices = [int(line.strip()) for line in f if line.strip()]
                if len(frame_indices) != len(poses):
                    raise ValueError(
                        f"poses_idx.txt count ({len(frame_indices)}) does not match poses count ({len(poses)})."
                    )
            else:
                frame_indices = list(range(len(poses)))

            from scipy.spatial.transform import Rotation

            tum_lines = []
            for i, pose in enumerate(poses):
                frame_idx = frame_indices[i]
                if frame_idx not in timestamps_by_frame:
                    raise ValueError(
                        f"Missing timestamp for frame index {frame_idx} in timestamps_by_frame."
                    )
                timestamp = timestamps_by_frame[frame_idx]

                # poses_est.txt is already C2W, so no inversion is needed here.
                tx, ty, tz = pose[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                tum_lines.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")

            tum_path = output_dir / "CameraTrajectory.txt"
            with open(tum_path, 'w') as f:
                f.write('\n'.join(tum_lines))

            logger.info(f"    Converted {len(tum_lines)} poses to TUM format")
            return tum_path

        except Exception as e:
            raise RuntimeError(f"Failed to convert GigaSLAM trajectory to TUM format: {e}") from e

    def cleanup(self) -> None:
        """Clean up temporary files and running processes."""
        # Kill any running GigaSLAM process
        if hasattr(self, '_current_process') and self._current_process:
            self._kill_process_group(self._current_process)
            self._current_process = None

        # Clean up temp config
        if self._temp_config and self._temp_config.exists():
            try:
                self._temp_config.unlink()
            except Exception as e:
                logger.warning(f"  Failed to cleanup temp config: {e}")
        self._temp_config = None
