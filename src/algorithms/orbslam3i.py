"""ORB-SLAM3 stereo-inertial (VIO) variant for SLAMAdversarialLab.

This runs the stock ORB-SLAM3 ``stereo_inertial_euroc`` driver from the
``orbslam3:latest`` image, i.e. ORB-SLAM3 in Visual-Inertial mode with IMU. It
exists so ORB-SLAM3 can be compared apples-to-apples against Nitro-SLAM, which
only runs on the EuRoC stereo-inertial path: same dataset, same sensor mode
(stereo + IMU), the difference being Nitro's GPU acceleration modules.

Unlike Nitro's ``stereo_inertial_euroc`` (which adds seven trailing module/kernel
arguments), the stock binary takes the plain ORB-SLAM3 signature
``<vocab> <settings> <sequence> <times_file> [trajectory_file_name]`` and, with no
file name, writes ``CameraTrajectory.txt`` / ``KeyFrameTrajectory.txt`` — so the
inherited ORB-SLAM3 trajectory collection applies unchanged. ORB-SLAM3 is
non-CUDA, so (like the base wrapper) HAMi GPU env is injected but inert.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from .base import ExecutionSpec
from .orbslam3 import ORBSLAM3Algorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"


class ORBSLAM3InertialAlgorithm(ORBSLAM3Algorithm):
    """Stock ORB-SLAM3 stereo-inertial (VIO) via the ``orbslam3:latest`` image."""

    @property
    def name(self) -> str:
        return "orbslam3i"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        # "stereo" here means EuRoC stereo-inertial; the wrapper always uses the
        # inertial binary and stages IMU data alongside the stereo cameras.
        return {
            "euroc": ["stereo"],
        }

    def resolve_config_name(
        self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None
    ) -> Optional[str]:
        if dataset_type.lower() == "euroc":
            return "EuRoC.yaml"
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        super()._preflight_checks(request, ctx)
        if request.dataset_type.lower() != "euroc":
            raise RuntimeError(
                f"{self.name} only supports EuRoC stereo-inertial; got '{request.dataset_type}'."
            )
        if self._resolve_euroc_imu_csv(request) is None:
            raise RuntimeError(
                f"{self.name} (stereo-inertial) requires EuRoC IMU data, but no "
                f"mav0/imu0/data.csv was found under {request.dataset_path}."
            )

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Stage EuRoC stereo structure (via ORB-SLAM3) plus the IMU stream.

        IMU resolve/stage helpers are shared with nitroslam and live on the
        ORBSLAM3Algorithm base.
        """
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

        executable = "./Examples/Stereo-Inertial/stereo_inertial_euroc"
        config_dir = "Examples/Stereo-Inertial"
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)

        docker_cmd = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{dataset_path.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
        ]

        if euroc_image_mounts:
            for host_path, container_path in euroc_image_mounts:
                docker_cmd.extend(["-v", f"{host_path}:{container_path}:ro"])

        if ctx.config_is_external:
            slam_config_resolved = ctx.resolved_config_path
            if slam_config_resolved is None:
                logger.error("External config path is not resolved in runtime context")
                return None
            config_abs_path = f"/orbslam3/{config_dir}/{slam_config_resolved.name}"
            docker_cmd.extend(["-v", f"{slam_config_resolved}:{config_abs_path}:ro"])
            logger.info("  Using external config: %s", slam_config_resolved)
        else:
            config_abs_path = f"{config_dir}/{ctx.internal_config_name}"
            logger.info("  Using internal config: %s", config_abs_path)

        # ORB-SLAM3 is non-CUDA: HAMi env/mounts/devices are injected (so the
        # runtime-stress plumbing matches other SLAMs) but inert.
        extras = self._runtime_stress_launch_extras()
        extra_env = extras.get("env") or {}
        extra_mounts = extras.get("mounts") or []
        extra_devices = extras.get("devices") or []
        extra_run_flags = list(extras.get("run_flags") or [])
        if extra_env or extra_mounts or extra_devices:
            logger.info(
                "  orbslam3i is non-CUDA; runtime-stress HAMi env vars are injected "
                "but inert — GPU caps have no effect."
            )
        for key, value in extra_env.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Launch-time flags (a constant memory cap applied at run rather than
        # by a later ``podman update``, which hung on at least one system).
        docker_cmd.extend(extra_run_flags)
        for src, dst, mode in extra_mounts:
            docker_cmd.extend(["-v", f"{src}:{dst}:{mode}"])
        for device in extra_devices:
            docker_cmd.extend(["--device", device])

        # Real-time deadline harness (in-loop frame dropping). The patched
        # stereo_inertial_euroc honors SAL_DEADLINE_* env; no-op unless the
        # pipeline set SAL_DEADLINE_FPS in the host environment.
        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd

        apply_realtime_to_podman_cmd(docker_cmd, _CONTAINER_OUTPUT_PATH)

        docker_cmd.append(self.docker_image)

        runtime_stress_active = ctx.runtime_stress is not None
        timestamps_file = f"{_CONTAINER_DATASET_PATH}/orbslam3_timestamps.txt"
        logger.info("  Using staged EuRoC timestamps file: %s", timestamps_file)

        # Stock signature: no file name => writes CameraTrajectory.txt /
        # KeyFrameTrajectory.txt (handled by the inherited bash command).
        slam_cmd = (
            f"xvfb-run -a {executable} Vocabulary/ORBvoc.txt {config_abs_path} "
            f"{_CONTAINER_DATASET_PATH} {timestamps_file}"
        )
        bash_cmd = self._build_container_bash_command(
            slam_cmd=slam_cmd,
            runtime_stress_active=runtime_stress_active,
        )
        docker_cmd.extend(["bash", "-c", bash_cmd])

        logger.info("  Executing ORB-SLAM3 (VI) via %s container: %s", self.container_runtime, container_name)
        io_target_paths = [str(dataset_path.resolve()), str(output_dir.resolve())]
        return ExecutionSpec(
            cmd=docker_cmd,
            stream_output=False,
            log_prefix="ORB-SLAM3-VI",
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
        )
