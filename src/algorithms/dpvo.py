"""DPVO / DPV-SLAM algorithm implementation.

DPVO (Deep Patch Visual Odometry, princeton-vl, MIT) is a monocular,
patch-based deep VO system from the DROID-SLAM lab: faster and lighter than
DROID, same CLI shape (``demo.py --imagedir --calib --stride``). DPV-SLAM is
its loop-closure extension in the same repo, enabled by
``--opts LOOP_CLOSURE True``; SAL registers it as the separate algorithm
``dpvslam`` (same image, one extra flag) for the VO-vs-SLAM contrast.

Podman-only wrapper (no conda path): ``container_runtime=None`` defaults to
podman; "docker" is accepted too (the docker CLI on this host talks to
podman's compat socket). Mono TUM only, mirroring DROID-SLAM's scope so the
two are directly comparable.

Trajectory: demo.py's ``--save_trajectory`` writes a TUM-format file via
evo, but its column-1 stamps are DPVO's SEQUENTIAL frame counters (the
``enumerate`` counter over the stride-sampled image list, verified from
``dpvo/stream.py``), not seconds. Host-side conversion maps counters through
the deadline drop log (identity when absent) to sampled-stream positions and
then to real TUM seconds, exactly like DROID's reconstruction conversion.
"""

import logging
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .droidslam import TUM_CALIBRATIONS
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_PODMAN_IMAGE = "dpvo:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
_CONTAINER_CALIB_PATH = "/calib/calib.txt"
_CONTAINER_WORKDIR = "/dpvo"
# demo.py writes saved_trajectories/<name>.txt relative to its cwd; the
# container command copies it to this fixed path in the output mount.
_RAW_TRAJECTORY_NAME = "dpvo_traj_raw.txt"


class DPVOAlgorithm(SLAMAlgorithm):
    """DPVO (Deep Patch Visual Odometry) via Podman container (mono TUM)."""

    _container_name_prefix = "dpvo"
    _log_prefix = "DPVO"
    # DPV-SLAM subclass appends its loop-closure opts here.
    _extra_demo_args = ""

    def __init__(self, container_runtime: Optional[str] = None):
        # Podman-only wrapper: None defaults to podman so registry/pipeline
        # instantiation with no args yields a working algorithm.
        if container_runtime is None:
            container_runtime = "podman"
        if container_runtime not in {"docker", "podman"}:
            raise ValueError(
                f"container_runtime must be 'docker' or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.dpvo_path = (
            Path(__file__).resolve().parents[2] / "deps" / "slam-algorithms" / "DPVO"
        )
        self.docker_image = _PODMAN_IMAGE

    @property
    def runtime_stress_target_kind(self) -> str:
        return f"{self.container_runtime}_container"

    @property
    def name(self) -> str:
        return "dpvo"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "tum": ["mono"],
        }

    def resolve_config_name(
        self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        """DPVO uses DROID-style calibration files; detect freiburg version."""
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
        """DPVO uses calibration identifiers, not internal config file paths."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate the container runtime + image (warn-only on missing image)."""
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
                    "Container image '%s' not found locally. Build with: "
                    "cd %s && ./build.sh %s",
                    self.docker_image,
                    self.dpvo_path,
                    self.container_runtime,
                )
        except Exception as exc:
            logger.info("Skipping container image existence check: %s", exc)

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)
        return {
            "dataset_path": request.dataset_path,
            "slam_config": ctx.internal_config_name or request.slam_config,
            "output_dir": request.output_dir,
            "dataset_type": request.dataset_type,
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
        frame_image_paths = inputs.get("frame_image_paths") or []

        calib_file = self._prepare_calibration(dataset_path, slam_config, dataset_type)
        if not calib_file:
            return None

        image_dir = self._prepare_dataset(dataset_path, dataset_type, camera_paths)
        if not image_dir:
            return None

        # Stage only adapter-approved frames so DPVO's sorted-glob count
        # matches timestamps_by_frame exactly (DPVO globs the image dir in
        # dpvo/stream.py; extra files would shift the counter-to-timestamp
        # mapping). Same rationale and mechanism as DROID-SLAM.
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
        """Build the container execution spec that runs ``dpvo:latest``."""
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stride = self._resolve_stride(dataset_type)

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{image_dir.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            "-v", f"{calib_file.resolve()}:{_CONTAINER_CALIB_PATH}:ro",
        ]

        # Override the container's snapshot of the frame-loop file with the
        # host's current copy so SAL deadline hooks reach the SLAM without an
        # image rebuild. DPVO's frame loop is in dpvo/stream.py (the
        # image_stream producer process), not demo.py. Works because the
        # image installs DPVO editable (pip install -e .), so imports resolve
        # to /dpvo/dpvo/stream.py.
        from ..runtime_stress.podman_injection import apply_entrypoint_override
        apply_entrypoint_override(
            container_cmd,
            host_path=self.dpvo_path / "dpvo" / "stream.py",
            container_path=f"{_CONTAINER_WORKDIR}/dpvo/stream.py",
        )

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # DPVO always needs CUDA; attach the GPU via CDI (DROID precedent:
        # explicit CDI failure beats a raw-device fallback's confusing
        # in-container CUDA error).
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

        # SAL real-time deadline harness propagation (no-op when inactive).
        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        # When HAMi is active, pre-create + bind-mount the per-container
        # shared cache file so GpuHamiController can mutate the cap mid-run.
        hami_active = "LD_PRELOAD" in (extras.get("env") or {})
        if hami_active:
            from ..runtime_stress.hami_controller import (
                HAMI_SHARED_CACHE,
                prepare_hami_cache_file,
            )
            cache_host_path = prepare_hami_cache_file(container_name)
            container_cmd.extend(["-v", f"{cache_host_path}:{HAMI_SHARED_CACHE}:rw"])

        container_cmd.append(self.docker_image)

        # demo.py saves via evo to saved_trajectories/<name>.txt (cwd
        # relative); copy it into the output mount under a fixed name.
        extra_args = f" {self._extra_demo_args}" if self._extra_demo_args else ""
        main_cmd = (
            f"cd {_CONTAINER_WORKDIR} && "
            f"python demo.py "
            f"--imagedir {_CONTAINER_DATASET_PATH} "
            f"--calib {_CONTAINER_CALIB_PATH} "
            f"--network dpvo.pth "
            f"--stride {stride} "
            f"--name salrun "
            f"--save_trajectory{extra_args} && "
            f"cp saved_trajectories/salrun.txt {_CONTAINER_OUTPUT_PATH}/{_RAW_TRAJECTORY_NAME}"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info(
            "  Executing %s via %s container: %s",
            self._log_prefix, self.container_runtime, container_name,
        )

        io_target_paths = [str(image_dir.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix=self._log_prefix,
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
        """Build a unique podman-compliant container name for one run."""
        raw_name = (
            f"{self._container_name_prefix}-{ctx.sequence_name}-"
            f"{output_dir.name}-{uuid.uuid4().hex[:8]}"
        )
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
        maths (DPVO processes the stride-sampled stream)."""
        return self._resolve_stride(dataset_type)

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run {self._log_prefix}: {e}")
            return False

    # ------------------------------------------------------------------
    # Calibration / dataset prep (DROID-format calib, reused conventions)
    # ------------------------------------------------------------------

    def _prepare_calibration(
        self, dataset_path: Path, slam_config: str, dataset_type: str
    ) -> Optional[Path]:
        """Create a DPVO calibration file (fx fy cx cy [dist...], DROID format)."""
        logger.info("  Preparing calibration file...")

        if dataset_type != "tum":
            logger.error(f"  Unsupported dataset type for calibration: {dataset_type}")
            return None

        calib_params = None
        config_lower = str(slam_config).lower()
        if "tum1" in config_lower or "freiburg1" in config_lower:
            calib_params = TUM_CALIBRATIONS["freiburg1"]
        elif "tum2" in config_lower or "freiburg2" in config_lower:
            calib_params = TUM_CALIBRATIONS["freiburg2"]
        elif "tum3" in config_lower or "freiburg3" in config_lower:
            calib_params = TUM_CALIBRATIONS["freiburg3"]
        else:
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

        calib_file = dataset_path / "calib_dpvo.txt"
        with open(calib_file, "w") as f:
            f.write(calib_params + "\n")
        logger.info(f"    Created calibration file: {calib_file}")
        return calib_file

    def _prepare_dataset(
        self,
        dataset_path: Path,
        dataset_type: str,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Resolve the image directory from the dataset runtime contract."""
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

        DPVO globs the image directory in dpvo/stream.py at startup; extra
        files not in associations.txt would shift its sequential counters
        away from timestamps_by_frame. Staging only approved files keeps the
        two aligned (same rationale as DROID-SLAM).
        """
        staging = output_dir / "_staged_rgb"
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
            "    Staged %d adapter-approved frames -> %s",
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

    # ------------------------------------------------------------------
    # Trajectory conversion (counter stamps -> remap -> real TUM seconds)
    # ------------------------------------------------------------------

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        raw = request.output_dir / _RAW_TRAJECTORY_NAME
        if raw.exists():
            return raw
        logger.error("  No %s trajectory found in %s", self._log_prefix, request.output_dir)
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Rewrite DPVO's counter-stamped TUM file with real dataset seconds.

        demo.py writes TUM-layout lines whose column 1 is DPVO's sequential
        frame counter over the stride-sampled stream (enumerate in
        dpvo/stream.py). Map counter -> (drop-log remap, identity when no
        deadline run) -> sampled position -> real TUM timestamp, mirroring
        DROID's reconstruction conversion.
        """
        inputs = ctx.execution_inputs or {}
        timestamps_by_frame = inputs.get("timestamps_by_frame") or {}
        dataset_type = inputs.get("dataset_type", request.dataset_type)
        stride = self._resolve_stride(dataset_type)

        try:
            frame_count = len(timestamps_by_frame)
            sampled_timestamps = [
                timestamps_by_frame[idx]
                for idx in range(frame_count)
                if idx % stride == 0
            ]
            if not sampled_timestamps:
                raise ValueError(
                    f"No sampled timestamps for stride={stride} with {frame_count} frames."
                )

            from ..runtime_stress.deadline_remap import (
                load_drop_log,
                remap_internal_indices,
            )

            drop_log = load_drop_log(Path(request.output_dir))
            # Fail loud rather than silently identity-map. DPVO keys poses by a
            # survivor counter, so a missing drop log while the deadline was
            # active would misalign every pose against ground truth.
            if os.environ.get("SAL_DEADLINE_FPS") and drop_log is None:
                raise RuntimeError(
                    "Deadline harness was active (SAL_DEADLINE_FPS set) but no "
                    f"drop log was found in {request.output_dir}. DPVO's "
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

            counters: List[int] = []
            rows: List[List[str]] = []
            with open(raw_trajectory) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) != 8:
                        raise ValueError(f"Malformed trajectory line: {line!r}")
                    counters.append(int(float(parts[0])))
                    rows.append(parts[1:])

            sampled_indices = remap_internal_indices(counters, drop_log)

            tum_lines: List[str] = []
            for sampled_idx, pose in zip(sampled_indices, rows):
                if sampled_idx < 0 or sampled_idx >= len(sampled_timestamps):
                    raise ValueError(
                        f"{self._log_prefix} timestamp index {sampled_idx} out of range "
                        f"for sampled stream of length {len(sampled_timestamps)} "
                        f"(stride={stride})."
                    )
                ts = sampled_timestamps[sampled_idx]
                tum_lines.append(f"{ts} " + " ".join(pose))

            traj_path = request.output_dir / "CameraTrajectory.txt"
            with open(traj_path, "w") as f:
                f.write("\n".join(tum_lines))

            logger.info("    Converted %d poses to TUM format", len(tum_lines))
            return traj_path

        except Exception as e:
            logger.error(f"  Failed to convert {self._log_prefix} trajectory: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self) -> None:
        """Stop and remove any DPVO containers for the active runtime."""
        runtime = self.container_runtime
        try:
            result = subprocess.run(
                [runtime, "ps", "-a", "--filter",
                 f"name={self._container_name_prefix}-", "-q"],
                capture_output=True, text=True, timeout=10,
            )
            ids = [cid for cid in result.stdout.strip().split("\n") if cid]
            for cid in ids:
                subprocess.run([runtime, "stop", "-t", "2", cid],
                               capture_output=True, timeout=15)
                subprocess.run([runtime, "rm", "-f", cid],
                               capture_output=True, timeout=10)
        except Exception as e:
            logger.warning("  Could not clean %s containers: %s", self._log_prefix, e)


class DPVSLAMAlgorithm(DPVOAlgorithm):
    """DPV-SLAM: DPVO plus proximity loop closure (same repo/image).

    Identical wrapper; the demo command additionally passes
    ``--opts LOOP_CLOSURE True`` (merged into DPVO's yacs config). The
    classic DBoW2 backend (CLASSIC_LOOP_CLOSURE) is intentionally not used:
    it needs extra C++ deps and the proximity backend is the paper's
    headline configuration.
    """

    _container_name_prefix = "dpvslam"
    _log_prefix = "DPV-SLAM"
    _extra_demo_args = "--opts LOOP_CLOSURE True"

    @property
    def name(self) -> str:
        return "dpvslam"
