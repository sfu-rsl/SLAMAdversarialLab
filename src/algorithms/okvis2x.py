"""OKVIS2-X algorithm implementation for SLAMAdversarialLab.

OKVIS2-X (https://github.com/ethz-mrl/OKVIS2-X) is a stereo visual-inertial
SLAM from ETH MRL extending OKVIS2 with optional learned stereo depth
(LibTorch), GNSS, LiDAR, and volumetric submapping. SAL integrates the two
EuRoC-relevant modes as separate algorithms sharing one container image
(``okvis2x:latest``, see deps/slam-algorithms/okvis2x-docker/):

* ``okvis2x``   - plain visual-inertial app (``okvis_app_synchronous``).
  CPU-bound compute, but the USE_GPU image build makes LibTorch initialize a
  CUDA context at startup, so the GPU device is attached in both modes (an
  idle context in this one; VRAM caps have little to bite).
* ``okvis2xnn`` - stereo-depth-network app
  (``okvis2x_app_snetwork_synchronous``), which runs Unimatch-based depth
  inference through CUDA LibTorch and feeds dense depth into the estimator.
  Live GPU workload; HAMi VRAM caps bite on the CUDA working set
  (Nitro-SLAM precedent).

Both reuse the ORB-SLAM3 wrapper's EuRoC staging (temp mav0 root with
generated camN/data.csv + image bind-mounts) plus the IMU staging that the
stereo-inertial wrappers (orbslam3i, nitroslam) rely on: OKVIS2-X's
DatasetReader consumes exactly that raw EuRoC ASL layout. Unlike the
ORB-SLAM3 family, OKVIS2-X takes an output directory as a CLI argument and
writes ``okvis2-<mode>[-...]_trajectory.csv`` files there, so trajectory
collection converts those CSVs to a TUM-format ``CameraTrajectory.txt``
host-side (nanosecond timestamps, the EuRoC metrics convention).
"""

import csv
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from .base import ExecutionSpec
from .orbslam3 import ORBSLAM3Algorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_PODMAN_IMAGE = "okvis2x:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
# OKVIS2-X's DatasetReader takes the mav0/ folder (EuRoC ASL layout root).
_CONTAINER_DATASET_ARG = f"{_CONTAINER_DATASET_PATH}/mav0"
# SAL-owned config copies baked into the image (see okvis2x-docker/configs/).
_CONTAINER_CONFIG_DIR = "/okvis2x/sal_configs"


class OKVIS2XAlgorithm(ORBSLAM3Algorithm):
    """OKVIS2-X visual-inertial mode via Docker or Podman container.

    Supports: EuRoC (stereo-inertial) through ``okvis_app_synchronous``.
    Inherits EuRoC staging, container lifecycle, and runtime-stress plumbing
    from :class:`ORBSLAM3Algorithm`; adds IMU staging, the OKVIS command
    line, and CSV-to-TUM trajectory conversion.
    """

    _container_name_prefix = "okvis2x"
    _cleanup_image_substring = "okvis2x"
    # Overridden by the NN subclass.
    _executable = "./okvis_app_synchronous"
    _needs_se2_config = False
    _gpu_workload = False
    # In-container display setup prefix. The VI app runs bare headless (the
    # vendored config disables every display output). The NN app calls
    # cv::imshow unconditionally, so it needs an X server; a manually started
    # Xvfb with a fixed display works, while xvfb-run kills the app at
    # startup in this image (verified empirically, GPU or not) and must NOT
    # be used.
    _display_prefix = ""

    def __init__(self, container_runtime: str = "docker"):
        if container_runtime not in {"docker", "podman"}:
            raise ValueError(
                f"container_runtime must be 'docker' or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.docker_image = _PODMAN_IMAGE

    @property
    def name(self) -> str:
        return "okvis2x"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        # "stereo" here means EuRoC stereo-inertial; OKVIS2-X requires IMU
        # (imu_parameters.use is true in the vendored config).
        return {
            "euroc": ["stereo"],
        }

    def resolve_config_name(
        self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        # One vendored config per mode, shared by all EuRoC sequences
        # (calibration is sequence-independent for EuRoC machine hall/vicon).
        if dataset_type.lower() == "euroc":
            return "euroc_vi.yaml"
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate OKVIS2-X runtime dependencies (EuRoC stereo-inertial only)."""
        runtime = self.container_runtime
        if shutil.which(runtime) is None:
            raise RuntimeError(
                f"OKVIS2-X requires {runtime!r}, but it was not found in PATH."
            )

        image_check = subprocess.run(
            [runtime, "image", "inspect", self.docker_image],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if image_check.returncode != 0:
            raise RuntimeError(
                f"OKVIS2-X container image not found in {runtime}: {self.docker_image}. "
                "Build the image (deps/slam-algorithms/okvis2x-docker/build.sh) before "
                "running evaluation."
            )

        dataset_type = request.dataset_type.lower()
        if dataset_type != "euroc":
            raise RuntimeError(
                f"OKVIS2-X only supports EuRoC stereo-inertial; got dataset '{dataset_type}'."
            )

        # Stereo cameras + per-frame timestamps (reused ORB-SLAM3 contracts).
        self._require_camera_dirs(request, ["left", "right"])
        self._require_camera_paths(request, ["left", "right"])
        self._require_timestamps_by_frame(request)

        # IMU is mandatory: the visual-inertial estimator integrates it.
        if self._resolve_euroc_imu_csv(request) is None:
            raise RuntimeError(
                "OKVIS2-X requires EuRoC IMU data, but no mav0/imu0/data.csv "
                f"was found under the dataset path {request.dataset_path}."
            )

    def _resolve_euroc_imu_csv(self, request: SLAMRunRequest) -> Optional[Path]:
        """Resolve the EuRoC IMU CSV, falling back to the original dataset.

        Perturbed EuRoC roots carry only the camera streams (SAL perturbs
        images, never IMU), so for perturbed runs the standard candidates
        under ``request.dataset_path`` do not exist. The original sequence
        root is recovered from ``extras['frame_image_paths']`` (absolute
        original left-camera image paths: .../<root>/mav0/cam0/data/x.png)
        and its pristine ``mav0/imu0/data.csv`` is used instead.
        """
        resolved = super()._resolve_euroc_imu_csv(request)
        if resolved is not None:
            return resolved

        frame_image_paths = request.extras.get("frame_image_paths") or []
        if not frame_image_paths:
            return None
        first = Path(str(frame_image_paths[0]))
        for parent in first.parents:
            candidate = parent / "mav0" / "imu0" / "data.csv"
            if candidate.exists():
                logger.info(
                    "  Using pristine IMU stream from the original dataset: %s",
                    candidate,
                )
                return candidate
        return None

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Stage EuRoC stereo structure (via ORB-SLAM3) plus the IMU stream."""
        staged_path = super()._stage_dataset(request, ctx)
        if staged_path is not None and request.dataset_type.lower() == "euroc":
            self._stage_euroc_imu(request, staged_path)
        return staged_path

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        dataset_path = inputs["dataset_path"]
        output_dir = inputs["output_dir"]
        euroc_image_mounts = inputs["euroc_image_mounts"]
        output_dir.mkdir(parents=True, exist_ok=True)

        container_name = self._build_runtime_stress_container_name(ctx, output_dir)

        container_cmd = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{dataset_path.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
        ]

        # Mount perturbed stereo image directories at the EuRoC-expected paths.
        if euroc_image_mounts:
            for host_path, container_path in euroc_image_mounts:
                container_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])
                logger.debug("  Mounting image dir: %s -> %s", host_path, container_path)

        # Config: the vendored in-image copy, or an external file bind-mounted
        # over the same directory so container paths stay uniform.
        if ctx.config_is_external:
            slam_config_resolved = ctx.resolved_config_path
            if slam_config_resolved is None:
                logger.error("External config path is not resolved in runtime context")
                return None
            config_abs_path = f"{_CONTAINER_CONFIG_DIR}/{slam_config_resolved.name}"
            container_cmd.extend(["-v", f"{slam_config_resolved}:{config_abs_path}:ro"])
            logger.info("  Using external config: %s", slam_config_resolved)
        else:
            config_abs_path = f"{_CONTAINER_CONFIG_DIR}/{ctx.internal_config_name}"
            logger.info("  Using internal config: %s", config_abs_path)

        # Runtime-stress / HAMi injection. The image is built USE_GPU=ON, so
        # LibTorch initializes a CUDA context at startup in BOTH modes and the
        # GPU device must always be attached (verified: without it, c10 aborts
        # with "CUDA driver version is insufficient" before streaming starts).
        # The VI app then leaves the context idle (no network inference), so
        # VRAM caps have little to bite there; the NN mode is the live CUDA
        # workload where HAMi caps are meaningful.
        extras = self._runtime_stress_launch_extras()
        extra_env = extras.get("env") or {}
        extra_mounts = extras.get("mounts") or []
        extra_devices = list(extras.get("devices") or [])
        extra_run_flags = list(extras.get("run_flags") or [])

        if not self._gpu_workload and extra_env:
            logger.info(
                "  okvis2x (VI mode) holds an idle CUDA context; HAMi caps have "
                "little to bite. Use okvis2xnn for the live GPU workload."
            )

        for key, value in extra_env.items():
            container_cmd.extend(["-e", f"{key}={value}"])
        for mount in extra_mounts:
            src, dst, mode = mount
            container_cmd.extend(["-v", f"{src}:{dst}:{mode}"])

        # Attach the GPU (both modes; see above). HAMi controls add
        # env/mounts on top but do not gate device access. The CDI device
        # syntax is used for BOTH runtimes: on this host the docker CLI talks
        # to podman's docker-compat socket (DOCKER_HOST), where --gpus all is
        # silently ignored (verified: c10 aborts with no GPU); CDI works
        # through both podman and docker>=25.
        if extra_devices:
            for device in extra_devices:
                container_cmd.extend(["--device", device])
        else:
            container_cmd.extend(["--device", "nvidia.com/gpu=all"])

        # Launch-time flags (a constant memory cap applied at run rather than by
        # a later ``podman update``). UNCONDITIONAL on purpose: this previously
        # sat inside the `if extra_devices:` branch above, so the cap was applied
        # only when GPU devices were present and a memory-only config silently
        # got none. That produced three okvis2x runs on the mid-run pathway the
        # flag exists to avoid, on the very system whose hang motivated it.
        container_cmd.extend(extra_run_flags)

        # When HAMi is active (LD_PRELOAD set), pre-create a per-container
        # host-side cache file so GpuHamiController.apply() can mutate the
        # cap mid-run. Mirrors Nitro-SLAM / VGGT-SLAM.
        if "LD_PRELOAD" in extra_env:
            from ..runtime_stress.hami_controller import (
                HAMI_SHARED_CACHE,
                prepare_hami_cache_file,
            )
            cache_host_path = prepare_hami_cache_file(container_name)
            container_cmd.extend(["-v", f"{cache_host_path}:{HAMI_SHARED_CACHE}:rw"])

        # Real-time deadline harness (in-loop frame dropping). The patched
        # DatasetReader honors SAL_DEADLINE_* env; no-op unless the pipeline
        # set SAL_DEADLINE_FPS in the host environment.
        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd

        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        container_cmd.append(self.docker_image)

        runtime_stress_active = ctx.runtime_stress is not None

        # App CLI (verified against the built binary): config-yaml [se2-yaml]
        # dataset-folder output-folder. The output-folder argument makes the
        # trajectory CSVs land directly in /output; no in-container copies.
        # Display handling comes from the per-mode _display_prefix (see the
        # class attribute docstring; xvfb-run must not be used).
        se2_arg = f"{_CONTAINER_CONFIG_DIR}/se2_euroc.yaml " if self._needs_se2_config else ""
        slam_cmd = (
            f"{self._display_prefix}{self._executable} {config_abs_path} {se2_arg}"
            f"{_CONTAINER_DATASET_ARG} {_CONTAINER_OUTPUT_PATH}"
        )

        bash_cmd = self._build_okvis_bash_command(
            slam_cmd=slam_cmd,
            runtime_stress_active=runtime_stress_active,
        )
        container_cmd.extend(["bash", "-c", bash_cmd])

        logger.info(
            "  Executing OKVIS2-X (%s) via %s container: %s",
            self._executable, self.container_runtime, container_name,
        )

        io_target_paths = [str(dataset_path.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix=self.name.upper(),
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
        )

    def _build_okvis_bash_command(
        self,
        *,
        slam_cmd: str,
        runtime_stress_active: bool,
    ) -> str:
        """In-container shell command for one OKVIS2-X run.

        OKVIS2-X writes its trajectory CSVs directly into /output (CLI arg),
        so unlike the ORB-SLAM3-family shapes there are no copy pairs; the
        trailing ``ls`` checks for the OKVIS CSVs to keep the log truthful.
        """
        traj_check = (
            "ls /output/okvis2-*_trajectory.csv 2>/dev/null || "
            "echo 'No OKVIS2-X trajectory CSVs generated'"
        )
        if not runtime_stress_active:
            return f"{slam_cmd}; {traj_check}"

        return (
            "set +e; "
            f"{slam_cmd} 2>&1 | tee /output/slam_output.log; "
            'slam_status=${PIPESTATUS[0]}; '
            "printf '%s\\n' \"$slam_status\" > /output/slam_exit_code.txt; "
            f"{traj_check}; "
            'exit "$slam_status"'
        )

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find the OKVIS2-X trajectory CSV in the run output directory.

        Prefers the final (loop-closed / final-BA) trajectory over the causal
        online one, mirroring how the ORB-SLAM3 family prefers the full
        post-Shutdown trajectory.
        """
        output_dir = request.output_dir
        # Most-optimized first: the final-BA trajectory (written after the
        # full final bundle adjustment, do_final_ba: true), then the
        # loop-closed final trajectory.
        final_ba_csvs = sorted(output_dir.glob("okvis2-*final-ba_trajectory.csv"))
        if final_ba_csvs:
            logger.info("  OKVIS2-X final-BA trajectory: %s", final_ba_csvs[0])
            return final_ba_csvs[0]

        final_csvs = sorted(output_dir.glob("okvis2-*final_trajectory.csv"))
        if final_csvs:
            logger.info("  OKVIS2-X final trajectory: %s", final_csvs[0])
            return final_csvs[0]

        online_csvs = [
            p for p in sorted(output_dir.glob("okvis2-*_trajectory.csv"))
            if "final" not in p.name
        ]
        if online_csvs:
            logger.warning(
                "  Only the online (causal) OKVIS2-X trajectory is available "
                "(final trajectory missing, possible early termination): %s",
                online_csvs[0],
            )
            return online_csvs[0]

        logger.error("  No OKVIS2-X trajectory CSVs found in %s", output_dir)
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert an OKVIS2-X trajectory CSV to TUM format.

        EuRoC metrics matching requires nanosecond timestamps in the first
        column (src/metrics/trajectory.py scales the evo association window
        by 1e9 when it sees ns-magnitude stamps), and OKVIS CSV timestamps
        are EuRoC-native nanoseconds, so they pass through unchanged.

        The CSV header is parsed defensively by column-name tokens rather
        than fixed positions: OKVIS writes timestamp, position, quaternion,
        and additional state columns (velocities, biases) that are ignored.
        """
        output_path = request.output_dir / "CameraTrajectory.txt"
        try:
            rows = self._parse_okvis_trajectory_csv(raw_trajectory)
        except ValueError as exc:
            logger.error("  OKVIS2-X trajectory conversion failed: %s", exc)
            return None

        if not rows:
            logger.error("  OKVIS2-X trajectory CSV has no data rows: %s", raw_trajectory)
            return None

        with open(output_path, "w", encoding="utf-8") as f:
            for ts, tx, ty, tz, qx, qy, qz, qw in rows:
                f.write(f"{ts} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

        logger.info(
            "  Converted OKVIS2-X trajectory to TUM: %s (%d poses)",
            output_path, len(rows),
        )
        return output_path

    @staticmethod
    def _parse_okvis_trajectory_csv(csv_path: Path) -> List[tuple]:
        """Parse an OKVIS trajectory CSV into (ts, tx..tz, qx..qw) tuples.

        Column mapping is by header-name token so extra columns (velocity,
        biases) and column reordering are tolerated. Raises ValueError with a
        descriptive message when the header cannot be mapped.
        """
        import re

        def _normalize(field: str) -> str:
            # "# p_WS_W_x [m]" -> "p_ws_w_x": drop comment marker and the
            # trailing unit bracket, keep the frame-annotated name.
            cleaned = field.strip().lstrip("#").strip().lower()
            return re.sub(r"\s*\[[^\]]*\]\s*$", "", cleaned).strip()

        def _find_column(fields: List[str], prefix: str, suffix: str) -> int:
            """Locate a column whose normalized name starts with ``prefix`` and
            ends with ``suffix`` (frame infixes like _WS_W_ are arbitrary).
            E.g. (prefix="p", suffix="_x") matches p_WS_W_x but not v_WS_W_x
            or b_g_x; (prefix="timestamp", suffix="") matches the stamp."""
            for idx, field in enumerate(fields):
                name = _normalize(field)
                if name.startswith(prefix) and name.endswith(suffix):
                    return idx
            raise ValueError(
                f"could not locate column ('{prefix}*{suffix}') in header "
                f"{fields!r} of {csv_path}"
            )

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                raise ValueError(f"empty trajectory CSV: {csv_path}")

            ts_idx = _find_column(header, "timestamp", "")
            # Position: p_WS_W_x style names (frame infix arbitrary).
            px_idx = _find_column(header, "p", "_x")
            py_idx = _find_column(header, "p", "_y")
            pz_idx = _find_column(header, "p", "_z")
            # Quaternion: q_WS_x style names. Written to TUM as x y z w.
            qx_idx = _find_column(header, "q", "_x")
            qy_idx = _find_column(header, "q", "_y")
            qz_idx = _find_column(header, "q", "_z")
            qw_idx = _find_column(header, "q", "_w")

            rows: List[tuple] = []
            for line in reader:
                if not line or (line[0].strip().startswith("#")):
                    continue
                try:
                    rows.append((
                        line[ts_idx].strip(),
                        float(line[px_idx]), float(line[py_idx]), float(line[pz_idx]),
                        float(line[qx_idx]), float(line[qy_idx]), float(line[qz_idx]),
                        float(line[qw_idx]),
                    ))
                except (IndexError, ValueError) as exc:
                    raise ValueError(
                        f"malformed trajectory row in {csv_path}: {line!r} ({exc})"
                    )
        return rows


class OKVIS2XNNAlgorithm(OKVIS2XAlgorithm):
    """OKVIS2-X stereo-depth-network mode (CUDA LibTorch) via container.

    Runs ``okvis2x_app_snetwork_synchronous``: Unimatch-based stereo depth
    inference feeds dense depth into the estimator alongside the sparse
    visual-inertial pipeline. A live GPU workload where HAMi VRAM caps bite.
    """

    _container_name_prefix = "okvis2xnn"
    _executable = "./okvis2x_app_snetwork_synchronous"
    _needs_se2_config = True
    _gpu_workload = True
    # The NN app calls cv::imshow unconditionally: start a fixed-display Xvfb
    # inline (xvfb-run kills the app at startup in this image; see base class).
    _display_prefix = (
        "Xvfb :99 -screen 0 1024x768x24 -nolisten tcp & sleep 1; DISPLAY=:99 "
    )

    @property
    def name(self) -> str:
        return "okvis2xnn"

    def resolve_config_name(
        self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        if dataset_type.lower() == "euroc":
            return "euroc_nn.yaml"
        return None
