"""VGGT-SLAM algorithm implementation."""

import logging
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_PODMAN_IMAGE = "vggtslam:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"


class VGGTSLAMAlgorithm(SLAMAlgorithm):
    """VGGT-SLAM: Dense RGB SLAM using VGGT model.

    VGGT-SLAM is a dense RGB-only SLAM system from MIT SPARK lab that uses
    the VGGT (Visual Geometry Grounded Transformer) model from Facebook.
    It supports monocular operation and is designed for accurate camera tracking.

    Supports: EuRoC (monocular)

    Reference: https://github.com/MIT-SPARK/VGGT-SLAM

    Two execution runtimes are supported:

    * ``container_runtime=None`` (default): run via the host ``vggt-slam``
      conda environment. This is the path used for everyday evaluation.
    * ``container_runtime="podman"``: run inside the ``vggtslam:latest``
      Podman container. This path is what runtime-stress scenarios use
      to exercise HAMi-style GPU caps against a real CUDA workload.
    """

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.vggtslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "VGGT-SLAM"
        self.conda_env = "vggt-slam"
        self.docker_image = _PODMAN_IMAGE

    @property
    def name(self) -> str:
        return "vggtslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "euroc": ["mono"],
            "tum": ["mono"],
        }

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        """VGGT-SLAM doesn't use config files or calibration; sequence name is passed through."""
        if dataset_type.lower() in ("euroc", "tum"):
            return sequence  # e.g., "V1_01_easy" or "freiburg1_desk"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """VGGT-SLAM does not use internal config file paths."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate VGGT-SLAM runtime dependencies."""
        if self.container_runtime is None:
            if not self.vggtslam_path.exists():
                raise RuntimeError(f"VGGT-SLAM not found at {self.vggtslam_path}")

            if not self._check_conda_available():
                raise RuntimeError(
                    f"Conda environment '{self.conda_env}' not found. "
                    f"Run install script in {self.vggtslam_path}."
                )
            return

        if shutil.which(self.container_runtime) is None:
            raise RuntimeError(
                f"VGGT-SLAM requires {self.container_runtime!r}, but it was not found in PATH."
            )

        if not self._container_image_exists():
            raise RuntimeError(
                f"Container image '{self.docker_image}' not found. "
                f"Build it with: cd {self.vggtslam_path} && "
                f"{self.container_runtime} build -t {self.docker_image} ."
            )

    def _container_image_exists(self) -> bool:
        """Return True if the configured container runtime has ``vggtslam:latest`` locally."""
        try:
            result = subprocess.run(
                [self.container_runtime, "image", "exists", self.docker_image],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
        except Exception as exc:
            logger.warning("Failed to probe %s image %s: %s", self.container_runtime, self.docker_image, exc)
            return False
        return result.returncode == 0

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Resolve the flat image folder VGGT-SLAM globs, for EuRoC or TUM mono.

        VGGT reads a folder via --image_folder and infers intrinsics from the
        VGGT model, so both datasets reduce to the dataset-resolved left-camera
        image directory (EuRoC mav0/cam0/...; TUM rgb/). Both are already
        truncated to max_frames by the pipeline's create_truncated_copy().
        """
        dataset_type = request.dataset_type.lower()
        camera_paths = request.extras.get("camera_paths", {})
        if dataset_type == "euroc":
            return self._prepare_euroc_dataset(request.dataset_path, camera_paths)
        if dataset_type == "tum":
            return self._prepare_tum_dataset(request.dataset_path, camera_paths)
        logger.error("  VGGT-SLAM does not support dataset type %r", request.dataset_type)
        return None

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

    def _prepare_tum_dataset(
        self,
        dataset_path: Path,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Resolve the TUM RGB image folder for VGGT-SLAM.

        TUM is monocular RGB; the pipeline resolves the left camera to the
        (already max_frames-truncated) rgb/ directory and passes it in
        request.extras['camera_paths']['left']. VGGT reads this folder directly
        via --image_folder and keys each pose by the float-second timestamp in
        the filename (e.g. 1305031452.791720.png), so no rgb.txt, no calib, and
        no per-frame staging are needed.
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
            logger.error(f"  TUM rgb directory not found: {image_dir}")
            return None

        image_files = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
        if not image_files:
            logger.error(f"  No PNG/JPG images found in {image_dir}")
            return None

        logger.info(f"  Using dataset-resolved TUM rgb images at {image_dir}")
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

        if self.container_runtime is None:
            return ExecutionSpec(
                cmd=["vggtslam"],
                custom_runner=lambda _spec, image_folder=image_folder, output_poses=output_poses: self._run_vggtslam(
                    image_folder,
                    output_poses,
                ),
                log_prefix="VGGT-SLAM",
            )

        output_dir = inputs["output_dir"]
        return self._build_container_execution_spec(ctx, image_folder, output_dir)

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        image_folder: Path,
        output_dir: Path,
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``vggtslam:latest`` against the dataset."""
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{image_folder.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
        ]

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # VGGT-SLAM always needs CUDA; ensure the GPU is always attached.
        # HAMi controls add extra env/mounts on top, but do not gate device access.
        devices = extras_devices if extras_devices else ["nvidia.com/gpu=all"]

        for key, value in (extras.get("env") or {}).items():
            container_cmd.extend(["-e", f"{key}={value}"])
        for mount in (extras.get("mounts") or []):
            src, dst, mode = mount
            container_cmd.extend(["-v", f"{src}:{dst}:{mode}"])
        for device in devices:
            container_cmd.extend(["--device", device])
        # Launch-time flags (e.g. a constant memory cap applied at run rather
        # than by a later update, which hung on at least one system).
        container_cmd.extend(extras.get("run_flags") or [])

        # When HAMi is active (LD_PRELOAD set), pre-create a per-container
        # host-side cache file and bind-mount it onto /tmp/cudevshr.cache so
        # GpuHamiController.apply() can mutate the cap mid-run by writing
        # directly to the mmap'd shared region. See
        # docs/fuzzy-slam/hami/HAMI_RUNTIME_MUTATION_INVESTIGATION.md.
        hami_active = "LD_PRELOAD" in (extras.get("env") or {})
        if hami_active:
            from ..runtime_stress.hami_controller import (
                HAMI_SHARED_CACHE,
                prepare_hami_cache_file,
            )
            cache_host_path = prepare_hami_cache_file(container_name)
            container_cmd.extend(
                ["-v", f"{cache_host_path}:{HAMI_SHARED_CACHE}:rw"]
            )

        # SAL real-time deadline harness: propagate the SAL_DEADLINE_* env +
        # bind-mount the iterator dir (no-op when the harness is off), and
        # override the image's baked-in main.py with the host copy so the
        # deadline hooks reach the SLAM without a rebuild.
        from ..runtime_stress.podman_injection import (
            apply_entrypoint_override,
            apply_realtime_to_podman_cmd,
        )
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)
        apply_entrypoint_override(
            container_cmd,
            host_path=Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "VGGT-SLAM" / "main.py",
            container_path="/vggt-slam/main.py",
        )

        container_cmd.append(self.docker_image)

        main_cmd = (
            f"cd /vggt-slam && "
            f"python main.py "
            f"--image_folder {_CONTAINER_DATASET_PATH} "
            f"--max_loops 1 "
            f"--conf_threshold 25 "
            f"--min_disparity 50 "
            f"--submap_size 16 "
            f"--log_results "
            f"--log_path {_CONTAINER_OUTPUT_PATH}/poses_raw.txt "
            f"--skip_dense_log"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info("  Executing VGGT-SLAM via %s container: %s", self.container_runtime, container_name)

        io_target_paths = [str(image_folder.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="VGGT-SLAM",
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
        )

    def _build_runtime_stress_container_name(
        self,
        ctx: SLAMRuntimeContext,
        output_dir: Path,
    ) -> str:
        """Build a unique container name for one VGGT-SLAM run."""
        raw_name = f"vggtslam-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

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
        dataset_type: str,
    ) -> Optional[Path]:
        """Convert VGGT-SLAM output to TUM format with dataset timestamps.

        VGGT-SLAM outputs: frame_id x y z qx qy qz qw, where frame_id is the
        number parsed from the image filename. For EuRoC the filename is the ns
        timestamp (an integer), mapped through timestamps_by_frame. For TUM the
        filename is already the float-second timestamp
        (e.g. 1305031452.791720.png), so column 0 IS the TUM timestamp and is
        passed through unchanged.

        Args:
            raw_poses_path: Path to VGGT-SLAM output
            output_dir: Output directory
            timestamps_by_frame: Frame-indexed timestamp mapping from run request
            dataset_type: "euroc" or "tum" -- selects the timestamp mapping

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

                    x, y, z = parts[1], parts[2], parts[3]
                    qx, qy, qz, qw = parts[4], parts[5], parts[6], parts[7]

                    if dataset_type.lower() == "tum":
                        # TUM frames are named by their float-second timestamp
                        # (1305031452.791720.png), which VGGT parses into column 0
                        # via re.search(r'\d+(?:\.\d+)?', name), so it already IS
                        # the TUM timestamp -- pass it through unchanged.
                        timestamp = parts[0]
                    else:
                        # EuRoC: the filename number is the ns timestamp (integer).
                        frame_id = int(float(parts[0]))
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
            request.dataset_type,
        )
