"""S3PO-GS (Gaussian Splatting SLAM) algorithm implementation."""

import json
import logging
import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


class S3POGSAlgorithm(SLAMAlgorithm):
    """S3PO-GS SLAM algorithm via conda environment. Monocular-only, KITTI datasets."""

    def __init__(self):
        self.s3pogs_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "S3PO-GS"
        self.conda_env = "S3PO-GS"
        self._temp_dataset_link: Optional[Path] = None

    @property
    def name(self) -> str:
        return "s3pogs"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "kitti": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        if dataset_type.lower() == "kitti":
            return str(sequence).zfill(2)
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return self.s3pogs_path / "configs" / "mono" / "KITTI" / f"{ctx.internal_config_name}.yaml"

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate S3PO-GS runtime dependencies."""
        if not self.s3pogs_path.exists():
            raise RuntimeError(f"S3PO-GS not found at {self.s3pogs_path}")

        config_file = self._resolve_config_path(ctx)

        if config_file is None:
            raise RuntimeError("S3PO-GS external config path is not resolved in runtime context")

        if not config_file.exists():
            raise RuntimeError(f"S3PO-GS config file not found: {config_file}")

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare dataset structure expected by S3PO-GS."""
        return self._prepare_dataset(
            request.dataset_path,
            ctx.sequence_name,
            request.extras.get("camera_paths", {}),
        )

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Remove temporary dataset staging directory."""
        self._cleanup_temp_dataset_link()

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by S3PO-GS execution."""
        prepared_path = ctx.effective_dataset_path
        config_file = self._resolve_config_path(ctx)
        if not prepared_path or config_file is None:
            return None
        return {
            "prepared_path": prepared_path,
            "output_dir": request.output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "config_file": config_file,
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        config_file = inputs["config_file"]

        return ExecutionSpec(
            cmd=["s3pogs"],
            custom_runner=lambda _spec, config_file=config_file: self._run_s3pogs(config_file),
            log_prefix="S3PO-GS",
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]
        config_file = inputs["config_file"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  S3PO-GS only supports monocular mode. Stereo flag ignored.")

        logger.info(f"  Using config: {config_file}")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            ctx.notes["execution_success"] = False
            return False
        try:
            ctx.notes["execution_started_at"] = time.time()
            execution_success = self._run_execution_spec(spec) == 0
            ctx.notes["execution_success"] = execution_success
            return execution_success
        except Exception as e:
            ctx.notes["execution_success"] = False
            logger.error(f"Failed to run S3PO-GS: {e}")
            return False

    def _prepare_dataset(
        self,
        dataset_path: Path,
        sequence: str,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Prepare dataset for S3PO-GS format (rgb/, poses.txt, calib.txt)."""
        logger.info("  Preparing dataset for S3PO-GS...")

        sequence_name = sequence.zfill(2)
        s3pogs_datasets = self.s3pogs_path / "datasets" / "KITTI"
        s3pogs_datasets.mkdir(parents=True, exist_ok=True)
        s3pogs_sequence = s3pogs_datasets / sequence_name

        if s3pogs_sequence.exists() or s3pogs_sequence.is_symlink():
            if s3pogs_sequence.is_symlink():
                s3pogs_sequence.unlink()
            else:
                shutil.rmtree(s3pogs_sequence)

        s3pogs_sequence.mkdir(parents=True, exist_ok=True)
        self._temp_dataset_link = s3pogs_sequence

        left_image_path = self._resolve_left_camera_path(camera_paths)
        if left_image_path is None:
            return None

        rgb_link = s3pogs_sequence / "rgb"

        try:
            os.symlink(str(left_image_path.resolve()), str(rgb_link))
        except (OSError, NotImplementedError) as e:
            logger.error(f"Failed to create symlink: {e}")
            return None

        # S3PO-GS expects poses.txt in KITTI format (12 values per line)
        poses_src = dataset_path / "poses.txt"
        poses_dst = s3pogs_sequence / "poses.txt"

        if poses_src.exists():
            shutil.copy2(poses_src, poses_dst)
        else:
            poses_dir = dataset_path / "poses"
            if poses_dir.exists():
                pose_files = list(poses_dir.glob("*.txt"))
                if pose_files:
                    shutil.copy2(pose_files[0].resolve(), poses_dst)
                else:
                    logger.error(f"No pose files found in {poses_dir}")
                    return None
            else:
                # Try multiple locations for KITTI poses
                kitti_poses_candidates = [
                    # Parent structure: ../../poses/{name}.txt
                    dataset_path.parent.parent / "poses" / f"{sequence_name}.txt",
                    # Project base datasets/kitti/poses/{name}.txt
                    Path(__file__).parent.parent.parent / "datasets" / "kitti" / "poses" / f"{sequence_name}.txt",
                ]

                poses_found = False
                for kitti_poses in kitti_poses_candidates:
                    if kitti_poses.exists():
                        shutil.copy2(kitti_poses, poses_dst)
                        logger.info(f"    Copied poses from: {kitti_poses}")
                        poses_found = True
                        break

                if not poses_found:
                    logger.error(f"poses.txt not found in {dataset_path} or any standard KITTI location")
                    return None

        calib_src = dataset_path / "calib.txt"
        if calib_src.exists():
            shutil.copy2(calib_src, s3pogs_sequence / "calib.txt")

        logger.info(f"    Dataset prepared at: {s3pogs_sequence}")
        return s3pogs_sequence

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

    def _run_s3pogs(self, config_file: Path) -> bool:
        """Execute S3PO-GS SLAM."""
        logger.info("  Executing S3PO-GS...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        cmd = (
            f"cd {self.s3pogs_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"python slam.py --config {config_file}"
        )

        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
                start_new_session=True  # Create new process group for clean termination
            )

            completed = False
            def _stop_on_line(line: str) -> bool:
                nonlocal completed
                if "Total FPS" in line:
                    completed = True
                    logger.info("  S3PO-GS evaluation complete, terminating process...")
                    return True
                return False

            self._stream_process_output(process, "S3PO-GS", stop_on_line=_stop_on_line)

            # Force terminate - S3PO-GS multiprocessing doesn't exit cleanly
            if completed:
                import signal
                try:
                    # Kill the process group to ensure all child processes are terminated
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                # Give it a moment then force kill if needed
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logger.info("  S3PO-GS completed successfully")
                return True

            self._wait_for_process(process, timeout_seconds=7200)

            if process.returncode != 0:
                logger.error(f"S3PO-GS failed with return code {process.returncode}")
                return False

            logger.info("  S3PO-GS completed successfully")
            return True

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("S3PO-GS timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run S3PO-GS: {e}")
            return False

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw S3PO-GS trajectory output."""
        if not ctx.notes.get("execution_success", False):
            logger.error("S3PO-GS execution did not complete successfully; skipping trajectory discovery")
            return None

        results_dir = self.s3pogs_path / "results"
        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return None

        run_started_at = float(ctx.notes.get("execution_started_at", 0.0))
        latest_run = None
        latest_time = None

        for dataset_dir in results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_dir in dataset_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                trj_file = run_dir / "plot" / "trj_final.json"
                if trj_file.exists():
                    mtime = trj_file.stat().st_mtime
                    if run_started_at and mtime + 1e-6 < run_started_at:
                        continue
                    if latest_time is None or mtime > latest_time:
                        latest_time = mtime
                        latest_run = trj_file

        if not latest_run:
            logger.error("No trajectory file found in S3PO-GS results")
            return None

        logger.info(f"    Found trajectory: {latest_run}")
        return latest_run

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert S3PO-GS trajectory JSON to TUM format with KITTI timestamps."""
        output_dir = request.output_dir
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)

        logger.info("  Converting trajectory to TUM format...")

        try:
            with open(raw_trajectory, 'r') as f:
                data = json.load(f)

            frame_ids = data["trj_id"]
            poses = data["trj_est"]

            # TUM format with actual KITTI timestamps
            # S3PO-GS outputs sparse keyframes - we map frame IDs to real timestamps
            tum_lines = []
            for frame_id, pose_4x4 in zip(frame_ids, poses):
                frame_idx = int(frame_id)
                if frame_idx not in timestamps_by_frame:
                    raise ValueError(
                        f"Missing timestamp for frame index {frame_idx} in timestamps_by_frame."
                    )
                timestamp = timestamps_by_frame[frame_idx]

                pose = np.array(pose_4x4)
                tx, ty, tz = pose[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                tum_lines.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")

            tum_path = output_dir / "CameraTrajectory.txt"
            with open(tum_path, 'w') as f:
                f.write('\n'.join(tum_lines))

            logger.info(f"    Converted {len(tum_lines)} keyframe poses to TUM format")
            return tum_path

        except Exception as e:
            logger.error(f"Failed to convert trajectory: {e}")
            return None

    def cleanup(self) -> None:
        """Remove temporary dataset symlink."""
        self._cleanup_temp_dataset_link()

    def _cleanup_temp_dataset_link(self) -> None:
        """Remove temporary S3PO-GS dataset directory/link when present."""
        if self._temp_dataset_link and self._temp_dataset_link.exists():
            try:
                if self._temp_dataset_link.is_symlink():
                    self._temp_dataset_link.unlink()
                else:
                    shutil.rmtree(self._temp_dataset_link)
            except Exception as e:
                logger.warning(f"  Failed to cleanup: {e}")

        self._temp_dataset_link = None
