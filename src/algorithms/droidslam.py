"""DROID-SLAM algorithm implementation.

DROID-SLAM: Deep Visual SLAM for Monocular, Stereo, and RGB-D Cameras.
Two execution runtimes are supported, mirroring VGGT-SLAM:

* ``container_runtime=None`` (default): host conda environment ``droidslam``.
* ``container_runtime="podman"``: ``droidslam:latest`` Podman container.
  This path is the one runtime-stress scenarios use to exercise HAMi GPU
  caps against DROID's CUDA-extension hot path.
"""

import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


# TUM freiburg calibration parameters (fx, fy, cx, cy, k1, k2, p1, p2, k3)
TUM_CALIBRATIONS = {
    "freiburg1": "517.3 516.5 318.6 255.3 0.2624 -0.9531 -0.0054 0.0026 1.1633",
    "freiburg2": "520.9 521.0 325.1 249.7 0.2312 -0.7849 -0.0033 -0.0001 0.9172",
    "freiburg3": "535.4 539.2 320.1 247.6",
}

_PODMAN_IMAGE = "droidslam:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
_CONTAINER_CALIB_PATH = "/calib/calib.txt"
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"
_CONTAINER_WORKDIR = "/droid-slam"


class DROIDSLAMAlgorithm(SLAMAlgorithm):
    """DROID-SLAM with dual conda+podman runtimes (mono TUM only today)."""

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.droid_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "DROID-SLAM"
        self.conda_env = "droidslam"
        self.docker_image = _PODMAN_IMAGE
        self._process = None

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

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
        if self.container_runtime is None:
            # Conda path: need the source tree and the in-tree weights.
            if not self.droid_path.exists():
                raise RuntimeError(f"DROID-SLAM not found at {self.droid_path}")
            weights_path = self.droid_path / "droid.pth"
            if not weights_path.exists():
                raise RuntimeError(
                    f"DROID-SLAM weights not found at {weights_path}. "
                    "Run: cd deps/slam-algorithms/DROID-SLAM && ./tools/download_model.sh"
                )
            return

        # Podman path: need the runtime + image. Don't fail on missing image —
        # warn so the operator can build it. The weights ship inside the image.
        if shutil.which(self.container_runtime) is None:
            raise RuntimeError(
                f"container_runtime is '{self.container_runtime}' but the binary "
                "is not on PATH. Install it or change container_runtime."
            )
        try:
            result = subprocess.run(
                [self.container_runtime, "image", "exists", self.docker_image],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            if result.returncode != 0:
                logger.warning(
                    "Podman image '%s' not found locally. Build with: "
                    "cd %s && podman build -t %s .",
                    self.docker_image,
                    self.droid_path,
                    self.docker_image,
                )
        except Exception as exc:
            logger.info(
                "Skipping podman image existence check: %s", exc,
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
            "frame_image_paths": request.extras.get("frame_image_paths") or [],
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
        frame_image_paths = inputs.get("frame_image_paths") or []

        # Prepare calibration file
        calib_file = self._prepare_calibration(dataset_path, slam_config, dataset_type)
        if not calib_file:
            return None

        # Prepare dataset path (find rgb directory)
        image_dir = self._prepare_dataset(dataset_path, dataset_type, camera_paths)
        if not image_dir:
            return None

        if self.container_runtime is None:
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

        # Stage only adapter-approved frames so DROID's os.listdir count
        # matches timestamps_by_frame exactly (avoids index-out-of-range
        # when the raw rgb/ dir has more files than associations.txt).
        if frame_image_paths:
            output_dir.mkdir(parents=True, exist_ok=True)
            image_dir = self._stage_approved_frames(output_dir, frame_image_paths)

        return self._build_container_execution_spec(
            ctx, image_dir, calib_file, output_dir, dataset_type
        )

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        image_dir: Path,
        calib_file: Path,
        output_dir: Path,
        dataset_type: str,
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``droidslam:latest``."""
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        stride = self._resolve_stride(dataset_type)

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{image_dir.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
            "-v", f"{calib_file.resolve()}:{_CONTAINER_CALIB_PATH}:ro",
        ]

        # Override the container's snapshot of demo.py with the host's
        # current copy so SAL hooks reach the SLAM without rebuilding.
        from ..runtime_stress.podman_injection import apply_entrypoint_override
        apply_entrypoint_override(
            container_cmd,
            host_path=Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "DROID-SLAM" / "demo.py",
            container_path=f"{_CONTAINER_WORKDIR}/demo.py",
        )

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # DROID-SLAM always needs CUDA; ensure the GPU is always attached.
        # Same CDI spec as every other GPU wrapper. If the host has no CDI
        # config, podman fails the launch with an explicit 'CDI device not
        # found' error — preferred over any raw-device fallback, which
        # attaches nodes without the driver user-space libs and turns the
        # infrastructure problem into a confusing in-container CUDA error.
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

        # SAL real-time deadline harness propagation (no-op when the
        # harness is not active in the host env).
        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        # When HAMi is active, pre-create + bind-mount the per-container shared
        # cache file so GpuHamiController.apply() can mutate the cap mid-run.
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

        container_cmd.append(self.docker_image)

        main_cmd = (
            f"cd {_CONTAINER_WORKDIR} && "
            f"python demo.py "
            f"--imagedir {_CONTAINER_DATASET_PATH} "
            f"--calib {_CONTAINER_CALIB_PATH} "
            f"--weights droid.pth "
            f"--stride {stride} "
            f"--disable_vis "
            f"--reconstruction_path {_CONTAINER_OUTPUT_PATH}/reconstruction.pth"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info(
            "  Executing DROID-SLAM via %s container: %s",
            self.container_runtime, container_name,
        )

        io_target_paths = [str(image_dir.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="DROID-SLAM",
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
        """Build a unique podman-compliant container name for one DROID-SLAM run."""
        raw_name = f"droidslam-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

    def _resolve_stride(self, dataset_type: str) -> int:
        """Frame stride. TUM is 1: every frame, so the deadline does all the dropping.

        It was 2, and that silently made the real-time deadline TWICE AS STRICT
        for this system than for any other. The deadline iterator wraps whatever
        list it is given and paces item k at k/target_fps, never consulting the
        stride, so 250 every-other frames whose real capture spans 16.6 s were
        handed out over 8.3 s. The eight stride-1 systems matched real capture
        time to within 0.1 s; only the two that subsampled were wrong, and by
        exactly the stride. Reading every frame removes the mismatch at the
        source rather than patching the pacing.

        The non-TUM branch is unchanged: these systems are not run elsewhere.
        """
        return 1 if dataset_type == "tum" else 3

    def frame_stride(self, dataset_type: str) -> int:
        """Public stride accessor for the deadline warmup-cutoff/completeness
        maths (DROID processes the stride-sampled stream)."""
        return self._resolve_stride(dataset_type)

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

    def _stage_approved_frames(
        self,
        output_dir: Path,
        frame_image_paths: List[str],
    ) -> Path:
        """Hardlink only adapter-approved RGB frames into a staging directory.

        DROID enumerates the image directory with os.listdir at startup.
        If the raw rgb/ folder contains extra files not in associations.txt,
        its listdir count diverges from timestamps_by_frame, causing an
        index-out-of-range crash in _convert_reconstruction. Staging only
        the approved files makes the two counts agree.
        """
        staging = output_dir / "_staged_rgb"
        # Rebuild from scratch every run. A reused output dir must not keep
        # hardlinks from a previous run: stale links can point at old image
        # content (a regenerated perturbed sequence keeps filenames but gets
        # new inodes) or include frames no longer in the approved set, either
        # of which makes DROID evaluate data this run did not request.
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for src_str in frame_image_paths:
            src = Path(src_str)
            dst = staging / src.name
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        logger.info(
            "    Staged %d adapter-approved frames → %s",
            len(frame_image_paths),
            staging,
        )
        return staging

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

        stride = self._resolve_stride(dataset_type)

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

            # When the SAL deadline harness was active, the DROID demo
            # processed only a subset of the sampled stream. The drop
            # log records which positions in the sampled stream were
            # actually consumed; remap DROID's sequential tstamps to
            # those positions before looking up real timestamps.
            from ..runtime_stress.deadline_remap import (
                load_drop_log,
                remap_internal_indices,
            )

            drop_log = load_drop_log(Path(output_dir))
            # Fail loud rather than silently identity-map. DROID keys poses by a
            # survivor counter, so a missing drop log while the deadline was
            # active would misalign every pose against ground truth.
            if os.environ.get("SAL_DEADLINE_FPS") and drop_log is None:
                raise RuntimeError(
                    "Deadline harness was active (SAL_DEADLINE_FPS set) but no "
                    f"drop log was found in {output_dir}. DROID-SLAM's "
                    "counter->timestamp remap requires it; refusing to "
                    "identity-map (would misalign the trajectory). The SLAM "
                    "likely crashed before writing the log or could not write it."
                )
            if drop_log is not None:
                logger.info(
                    "  Deadline drop log found: %d survivors of %d sampled frames",
                    len(drop_log.survivors),
                    drop_log.total_items,
                )

            sampled_indices = remap_internal_indices(
                [int(raw_idx) for raw_idx in tstamps],
                drop_log,
            )

            real_timestamps: List[float] = []
            for sampled_idx in sampled_indices:
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
        """Find trajectory file in output directory.

        Conda path: ``_run_droidslam`` converts the reconstruction inline,
        so ``CameraTrajectory.txt`` is the result.

        Podman path: the container only writes ``reconstruction.pth``;
        host-side conversion runs in ``_convert_raw_trajectory_to_tum``.
        """
        camera_traj = output_dir / "CameraTrajectory.txt"
        if camera_traj.exists():
            return camera_traj
        reconstruction = output_dir / "reconstruction.pth"
        if reconstruction.exists():
            return reconstruction
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
        """If the raw output is the conda path's already-converted TUM file,
        return it as-is. If it's the Podman path's reconstruction.pth, convert
        on the host now (mirroring the conda path's inline conversion)."""
        if raw_trajectory.suffix.lower() != ".pth":
            return raw_trajectory

        inputs = ctx.execution_inputs or {}
        dataset_type = inputs.get("dataset_type")
        timestamps_by_frame = inputs.get("timestamps_by_frame")
        output_dir = inputs.get("output_dir") or raw_trajectory.parent

        if not dataset_type or not isinstance(timestamps_by_frame, dict):
            logger.error(
                "  Missing dataset_type / timestamps_by_frame in execution inputs; "
                "cannot convert reconstruction.pth to TUM."
            )
            return None

        stride = self._resolve_stride(dataset_type)
        ok = self._convert_reconstruction(
            raw_trajectory,
            Path(output_dir),
            dataset_type,
            stride,
            timestamps_by_frame,
        )
        if not ok:
            return None
        return Path(output_dir) / "CameraTrajectory.txt"

    def cleanup(self) -> None:
        """Clean up resources."""
        if self._process is not None:
            self._kill_process_group(self._process)
            self._process = None
