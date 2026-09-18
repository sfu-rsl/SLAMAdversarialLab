"""cuVSLAM (NVIDIA Isaac) via Podman, EuRoC stereo-inertial.

cuVSLAM ships a prebuilt ``libcuvslam.so`` plus nanobind Python bindings, not a
runnable pipeline: the library has no frame loop and no trajectory writer, so
the caller owns both. Our loop is in ``sal_driver.py``, next to the image
build context, which is what makes this wrapper straightforward rather than
awkward -- the driver is an ordinary Python file, so the deadline harness can be
bind-mounted over it exactly as for DPVO. A SLAM whose loop is compiled into the
image cannot be reached that way.

Subclasses :class:`ORBSLAM3Algorithm` for the EuRoC staging only, which is the
same thing :class:`OKVIS2XAlgorithm` does and for the same reason: staging a
stereo pair into ``mav0/camN/data`` with a generated ``data.csv``, plus the IMU
stream, is dataset work rather than SLAM work, and it is already correct there.
Nothing else is inherited in spirit -- the binary, container, command line and
trajectory format are all cuVSLAM's own.

WHAT THIS WRAPPER DOES NOT DO. It does not rectify or undistort. EuRoC ships raw
radial-tangential images and cuVSLAM undistorts internally when
``rectified_stereo_camera=False``, which is what the driver sets. Passing
pre-rectified images while leaving that flag false would be a silent accuracy
bug, so the dataset is handed over untouched.
"""

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import ExecutionSpec
from .orbslam3 import ORBSLAM3Algorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_PODMAN_IMAGE = "cuvslam:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
# The driver reads the EuRoC sensor root, one level below the staged dataset.
_CONTAINER_DATASET_ARG = "/dataset/mav0"
_CONTAINER_DRIVER_PATH = "/cuvslam/sal_driver.py"
_TRAJECTORY_NAME = "CameraTrajectory.txt"


class CuVSLAMAlgorithm(ORBSLAM3Algorithm):
    """cuVSLAM stereo-inertial on EuRoC, container-only."""

    _container_name_prefix = "cuvslam"
    _cleanup_image_substring = "cuvslam"
    # Live CUDA workload: cuVSLAM's tracker runs GFTT, LK and bundle adjustment
    # on the GPU for every frame, so VRAM caps have something to bite (unlike
    # okvis2x's idle context).
    _gpu_workload = True
    # Which cuVSLAM odometry mode the driver selects. "inertial" is stereo+IMU,
    # cuVSLAM's own default for EuRoC.
    _odometry_mode = "inertial"

    def __init__(self, container_runtime: Optional[str] = None):
        # Defaults to podman because the image is built for the runtime-stress
        # path. The kwarg name is load-bearing: the registry instantiates with
        # no arguments, and `_reconcile_algorithm_runtime` reconstructs with
        # `type(algorithm)(container_runtime=...)` when the YAML disagrees.
        runtime = container_runtime or "podman"
        if runtime not in {"docker", "podman"}:
            raise ValueError(
                f"container_runtime must be 'docker' or 'podman', got {runtime!r}"
            )
        self.container_runtime = runtime
        self.docker_image = _PODMAN_IMAGE
        self.cuvslam_driver_path = (
            Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "cuvslam-docker" / "sal_driver.py"
        )

    @property
    def name(self) -> str:
        return "cuvslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        # "stereo" is EuRoC stereo-inertial. cuVSLAM has no mono-inertial mode,
        # and its Mono mode is scale-free, so IMU is what makes this metric.
        return {"euroc": ["stereo"]}

    def resolve_config_name(
        self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        # Identifier, not a file. Calibration comes from the dataset's own
        # sensor.yaml at run time, so there is no per-sequence config to pick.
        if dataset_type.lower() == "euroc":
            return f"cuvslam_euroc_{self._odometry_mode}"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        # Not file-backed: cuVSLAM is configured through constructor arguments
        # in the driver, and the rig is read from the staged sensor.yaml.
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        runtime = self.container_runtime
        if shutil.which(runtime) is None:
            raise RuntimeError(f"cuVSLAM requires {runtime!r}, but it was not found in PATH.")

        image_check = subprocess.run(
            [runtime, "image", "inspect", self.docker_image],
            capture_output=True, text=True, check=False, timeout=30,
        )
        if image_check.returncode != 0:
            raise RuntimeError(
                f"cuVSLAM container image not found in {runtime}: {self.docker_image}. "
                "Build it from deps/slam-algorithms/cuvslam-docker/ before evaluating."
            )

        dataset_type = request.dataset_type.lower()
        if dataset_type != "euroc":
            raise RuntimeError(
                f"cuVSLAM is wired for EuRoC stereo-inertial only; got '{dataset_type}'."
            )

        # Stereo pair + per-frame timestamps, reusing the inherited contracts.
        self._require_camera_dirs(request, ["left", "right"])
        self._require_camera_paths(request, ["left", "right"])
        self._require_timestamps_by_frame(request)

        if self._odometry_mode == "inertial" and self._resolve_euroc_imu_csv(request) is None:
            raise RuntimeError(
                "cuVSLAM stereo-inertial requires EuRoC IMU data, but no "
                f"mav0/imu0/data.csv was found under {request.dataset_path}."
            )

        if not self.cuvslam_driver_path.exists():
            raise RuntimeError(
                f"cuVSLAM driver missing at {self.cuvslam_driver_path}. It is "
                "bind-mounted over the in-image copy so host edits take effect; "
                "without it the container would silently run a stale driver."
            )

    def _stage_dataset(
        self, request: SLAMRunRequest, ctx: SLAMRuntimeContext
    ) -> Optional[Path]:
        """EuRoC stereo structure (inherited) plus the IMU stream."""
        staged_path = super()._stage_dataset(request, ctx)
        if staged_path is not None and request.dataset_type.lower() == "euroc":
            self._stage_euroc_imu(request, staged_path)
        return staged_path

    def _build_execution_inputs(
        self, request: SLAMRunRequest, ctx: SLAMRuntimeContext
    ) -> Optional[Dict[str, Any]]:
        return {
            "dataset_path": ctx.effective_dataset_path or request.dataset_path,
            "output_dir": request.output_dir,
            "euroc_image_mounts": ctx.staging_artifacts.get("euroc_image_mounts") or [],
        }

    def _build_execution_spec(
        self, request: SLAMRunRequest, ctx: SLAMRuntimeContext
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        dataset_path: Path = inputs["dataset_path"]
        output_dir: Path = inputs["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        container_name = self._build_runtime_stress_container_name(ctx, output_dir)

        container_cmd = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{dataset_path.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
        ]

        # Perturbed stereo image directories at the EuRoC-expected paths.
        for host_path, container_path in inputs["euroc_image_mounts"]:
            container_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])

        # The driver is bind-mounted over its in-image copy. Without this the
        # container runs the driver baked in at build time, so a host edit --
        # including the deadline wiring -- silently does nothing.
        from ..runtime_stress.podman_injection import apply_entrypoint_override

        apply_entrypoint_override(
            container_cmd,
            host_path=self.cuvslam_driver_path,
            container_path=_CONTAINER_DRIVER_PATH,
        )

        extras = self._runtime_stress_launch_extras()
        extra_env = extras.get("env") or {}
        extra_mounts = extras.get("mounts") or []
        extra_devices = list(extras.get("devices") or [])
        extra_run_flags = list(extras.get("run_flags") or [])

        for key, value in extra_env.items():
            container_cmd.extend(["-e", f"{key}={value}"])
        for src, dst, mode in extra_mounts:
            container_cmd.extend(["-v", f"{src}:{dst}:{mode}"])

        # CDI device syntax for both runtimes; cuVSLAM aborts without a GPU.
        if extra_devices:
            for device in extra_devices:
                container_cmd.extend(["--device", device])
        else:
            container_cmd.extend(["--device", "nvidia.com/gpu=all"])

        # Launch-time flags (e.g. a constant --memory cap). UNCONDITIONAL: the
        # okvis2x comment records what happens when this sits inside the device
        # branch -- memory-only configs silently get no cap at all.
        container_cmd.extend(extra_run_flags)

        if "LD_PRELOAD" in extra_env:
            from ..runtime_stress.hami_controller import (
                HAMI_SHARED_CACHE,
                prepare_hami_cache_file,
            )
            cache_host_path = prepare_hami_cache_file(container_name)
            container_cmd.extend(["-v", f"{cache_host_path}:{HAMI_SHARED_CACHE}:rw"])

        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd

        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        container_cmd.append(self.docker_image)

        slam_cmd = (
            f"python3 {_CONTAINER_DRIVER_PATH}"
            f" --dataset {_CONTAINER_DATASET_ARG}"
            f" --output {_CONTAINER_OUTPUT_PATH}"
            f" --mode {self._odometry_mode}"
        )

        from .config_utils import tee_console_output

        container_cmd.extend(["bash", "-c", tee_console_output(slam_cmd, _CONTAINER_OUTPUT_PATH)])

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="CUVSLAM",
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": [
                    str(dataset_path.resolve()),
                    str(output_dir.resolve()),
                ],
            },
        )

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        return self._run_execution_spec(spec) == 0

    def _find_raw_trajectory(
        self, request: SLAMRunRequest, ctx: SLAMRuntimeContext
    ) -> Optional[Any]:
        traj = request.output_dir / _TRAJECTORY_NAME
        return traj if traj.exists() else None

    def _convert_raw_trajectory_to_tum(
        self, raw_trajectory: Any, request: SLAMRunRequest, ctx: SLAMRuntimeContext
    ) -> Optional[Path]:
        """Already TUM with nanosecond stamps; validated rather than rewritten.

        The driver emits `timestamp tx ty tz qx qy qz qw` with integer
        nanosecond timestamps, which is what EuRoC ground truth uses and what
        the metrics layer expects. Rewriting it here would only risk a
        precision round trip: EuRoC stamps are ~1.4e18 ns, beyond float64's
        exact-integer range.
        """
        traj_path = Path(raw_trajectory)
        if not traj_path.exists() or traj_path.stat().st_size == 0:
            logger.error("cuVSLAM produced no trajectory at %s", traj_path)
            return None
        return traj_path
