"""Nitro-SLAM algorithm implementation for SLAMAdversarialLab.

Nitro-SLAM (https://github.com/sfu-rsl/Nitro-SLAM) is an ORB-SLAM3 fork from
SFU RSL that adds four GPU acceleration modules: FastTrack (front-end tracking),
TurboMap (local mapping), FastLoop (loop closure), and Graphite (graph
optimization), built on CUDA 12.8 + Vulkan/Kompute.

Crucially, those GPU modules are only ever activated by the **stereo-inertial**
driver programs (``Examples/Stereo-Inertial/stereo_inertial_euroc.cc`` and the
TUM-VI equivalent): they call ``TrackingKernelController::activate()`` +
``setGPURunMode()``. Every other driver (KITTI mono/stereo, TUM RGB-D, the
non-inertial EuRoC binaries) leaves the kernel controllers at their default
``is_active=false`` state and runs as stock ORB-SLAM3 on CPU.

This wrapper therefore targets the one path where Nitro's GPU modules genuinely
engage (and where HAMi VRAM caps actually bite): **EuRoC stereo-inertial** via
``stereo_inertial_euroc``. It reuses the ORB-SLAM3 wrapper's EuRoC staging,
camera/timestamp validation, and trajectory handling, and layers on the
inertial-specific bits: IMU staging, the seven trailing module/kernel arguments,
and live CUDA GPU + HAMi injection (unlike ORB-SLAM3, which is CPU-only so its
HAMi plumbing is inert).
"""

import logging
from typing import Optional

from .base import ExecutionSpec
from .orbslam3i import ORBSLAM3InertialAlgorithm
from .types import SLAMRunRequest, SLAMRuntimeContext

logger = logging.getLogger(__name__)

_PODMAN_IMAGE = "nitroslam:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"

# Nitro-SLAM GPU module toggles. We run with all four acceleration modules
# enabled, using Nitro's documented default kernel bitmasks (see its README:
# FastTrack 11110 / TurboMap 1111 / FastLoop 11111). These are passed as the
# seven trailing arguments to ``stereo_inertial_euroc``:
#   ... <strStatsFile> <FastTrack> <TurboMap> <FastLoop> <FT_bm> <TM_bm> <FL_bm>
_MODULE_FASTTRACK = "1"
_MODULE_TURBOMAP = "1"
_MODULE_FASTLOOP = "1"
_KERNEL_BITMASK_FASTTRACK = "11110"
_KERNEL_BITMASK_TURBOMAP = "1111"
_KERNEL_BITMASK_FASTLOOP = "11111"

# stereo_inertial_euroc requires a trajectory_file_name argument: for a single
# sequence its arg-count guard (min_num_argc = 6 + 7) forces bFileName=true, and
# it then writes f_<name>.txt (full camera trajectory) and kf_<name>.txt
# (keyframe trajectory) to the working directory. We copy those to /output as
# Camera/KeyFrameTrajectory.txt so the inherited trajectory collection applies.
_TRAJECTORY_BASENAME = "nitro"


class NitroSLAMAlgorithm(ORBSLAM3InertialAlgorithm):
    """Nitro-SLAM (GPU-accelerated ORB-SLAM3 fork) via Docker or Podman container.

    Supports: EuRoC (stereo-inertial). Runs the ``stereo_inertial_euroc``
    driver, the only Nitro binary that activates the FastTrack/TurboMap/FastLoop
    GPU modules. Nitro is ORB-SLAM3 stereo-inertial plus GPU modules, so it
    subclasses :class:`ORBSLAM3InertialAlgorithm` and inherits its dataset
    support, config resolution, preflight (EuRoC gate + IMU check), IMU
    staging, trajectory handling, and container lifecycle. This wrapper adds
    only the Nitro-specific bits: the module/kernel arguments, the f_/kf_
    trajectory copy mapping, and always-on GPU + live HAMi injection.
    """

    _container_name_prefix = "nitroslam"
    _cleanup_image_substring = "nitroslam"
    _image_build_hint = (
        "Build the image (deps/slam-algorithms/nitroslam-docker/build.sh) "
        "before running evaluation."
    )

    def __init__(self, container_runtime: str = "docker"):
        super().__init__(container_runtime)
        self.docker_image = _PODMAN_IMAGE

    @property
    def name(self) -> str:
        return "nitroslam"

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
        logger.info("  Using Nitro-SLAM directory: %s", config_dir)

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

        # Settings: Nitro's bundled stereo-inertial config (container-relative).
        if ctx.config_is_external:
            slam_config_resolved = ctx.resolved_config_path
            if slam_config_resolved is None:
                logger.error("External config path is not resolved in runtime context")
                return None
            config_path_in_container = f"{config_dir}/{slam_config_resolved.name}"
            config_abs_path = f"/Nitro-SLAM/{config_path_in_container}"
            container_cmd.extend(["-v", f"{slam_config_resolved}:{config_abs_path}:ro"])
            logger.info("  Using external config: %s", slam_config_resolved)
        else:
            config_abs_path = f"{config_dir}/{ctx.internal_config_name}"
            logger.info("  Using internal config: %s", config_abs_path)

        # Runtime-stress / HAMi injection. Unlike ORB-SLAM3 (CPU-only, inert),
        # Nitro is a real CUDA workload, so these env/mounts/devices are live.
        extras = self._runtime_stress_launch_extras()
        extra_env = extras.get("env") or {}
        extra_mounts = extras.get("mounts") or []
        extra_devices = list(extras.get("devices") or [])
        extra_run_flags = list(extras.get("run_flags") or [])

        for key, value in extra_env.items():
            container_cmd.extend(["-e", f"{key}={value}"])

        # Launch-time flags (a constant memory cap applied at run rather than
        # by a later ``podman update``, which hung on at least one system).
        container_cmd.extend(extra_run_flags)
        for mount in extra_mounts:
            src, dst, mode = mount
            container_cmd.extend(["-v", f"{src}:{dst}:{mode}"])

        # Nitro always needs the GPU. HAMi controls add env/mounts on top but do
        # not gate device access; attach the GPU when the session did not already.
        if extra_devices:
            for device in extra_devices:
                container_cmd.extend(["--device", device])
        elif self.container_runtime == "podman":
            container_cmd.extend(["--device", "nvidia.com/gpu=all"])
        else:  # docker
            container_cmd.extend(["--gpus", "all"])

        # When HAMi is active (LD_PRELOAD set), pre-create a per-container
        # host-side cache file and bind-mount it onto /tmp/cudevshr.cache so
        # GpuHamiController.apply() can mutate the cap mid-run. Mirrors VGGT-SLAM.
        if "LD_PRELOAD" in extra_env:
            from ..runtime_stress.hami_controller import (
                HAMI_SHARED_CACHE,
                prepare_hami_cache_file,
            )
            cache_host_path = prepare_hami_cache_file(container_name)
            container_cmd.extend(["-v", f"{cache_host_path}:{HAMI_SHARED_CACHE}:rw"])

        # Real-time deadline harness (in-loop frame dropping). The patched
        # stereo_inertial_euroc honors SAL_DEADLINE_* env; no-op unless the
        # pipeline set SAL_DEADLINE_FPS in the host environment.
        from ..runtime_stress.podman_injection import apply_realtime_to_podman_cmd

        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        container_cmd.append(self.docker_image)

        runtime_stress_active = ctx.runtime_stress is not None

        timestamps_file = f"{_CONTAINER_DATASET_PATH}/orbslam3_timestamps.txt"
        logger.info("  Using staged EuRoC timestamps file: %s", timestamps_file)

        module_args = (
            f"{_MODULE_FASTTRACK} {_MODULE_TURBOMAP} {_MODULE_FASTLOOP} "
            f"{_KERNEL_BITMASK_FASTTRACK} {_KERNEL_BITMASK_TURBOMAP} {_KERNEL_BITMASK_FASTLOOP}"
        )
        # Arg order: vocabulary settings sequence times_file file_name
        # strStatsFile FastTrack TurboMap FastLoop ft_bitmask tm_bitmask fl_bitmask.
        # statsDir is /output; the f_/kf_ trajectory files land in the working dir.
        slam_cmd = (
            f"xvfb-run -a {executable} Vocabulary/ORBvoc.txt {config_abs_path} "
            f"{_CONTAINER_DATASET_PATH} {timestamps_file} {_TRAJECTORY_BASENAME} "
            f"{_CONTAINER_OUTPUT_PATH} {module_args}"
        )

        # stereo_inertial_euroc writes f_<name>.txt (full trajectory) and
        # kf_<name>.txt (keyframes) to the working dir; map them to the
        # standard /output names so the inherited trajectory collection
        # applies unchanged.
        bash_cmd = self._build_container_bash_command(
            slam_cmd=slam_cmd,
            runtime_stress_active=runtime_stress_active,
            copy_pairs=[
                (f"f_{_TRAJECTORY_BASENAME}.txt", "/output/CameraTrajectory.txt"),
                (f"kf_{_TRAJECTORY_BASENAME}.txt", "/output/KeyFrameTrajectory.txt"),
            ],
        )
        container_cmd.extend(["bash", "-c", bash_cmd])

        logger.info("  Executing Nitro-SLAM via %s container: %s", self.container_runtime, container_name)

        io_target_paths = [str(dataset_path.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="Nitro-SLAM",
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
        )
