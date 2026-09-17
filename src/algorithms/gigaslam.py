"""GigaSLAM (Large-Scale Monocular SLAM with Hierarchical Gaussian Splats).

GigaSLAM is a monocular Gaussian-splat SLAM that uses UniDepthV2 (foundation
monocular depth) for per-frame depth and ORB-SLAM3-style features + DBoW2 for
loop closure. Two execution runtimes are wired into the SAL framework,
mirroring VGGT-SLAM, DROID-SLAM, Photo-SLAM, and MASt3R-SLAM:

* ``container_runtime=None`` (default): host conda environment ``gigaslam``
  at ``deps/slam-algorithms/GigaSLAM`` invoked via ``python slam.py --config
  <generated.yaml> --eval``.
* ``container_runtime="podman"``: ``gigaslam:latest`` Podman image. This
  path is the one runtime-stress scenarios use to exercise HAMi GPU caps
  (including per-phase runtime mutation) against GigaSLAM's UniDepthV2 +
  Gaussian-splat + ``sim3solve`` (CUDAExtension) hot path. Third
  foundation-model SLAM ported to the dual-runtime contract (sibling to
  VGGT-SLAM and MASt3R-SLAM).
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

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


_PODMAN_IMAGE = "gigaslam:latest"
# GigaSLAM derives the save-subdir name from the LAST three components of
# ``Dataset.color_path`` (slam.py: ``path[-3]+"_"+path[-2]+"_"+path[-1]``).
# We mount the KITTI sequence root at /dataset/kitti/sequences/<NN>/ so that
# color_path = /dataset/kitti/sequences/<NN>/image_X produces a save-subdir
# that includes the sequence id, mirroring the conda-path behavior.
_CONTAINER_DATASET_PARENT = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
# UniDepthV2 weights are pulled from HuggingFace on first run via
# huggingface_hub.from_pretrained() (slam_frontend.py:99-103). Mount the
# host's HF cache so the weights persist across container invocations and
# don't get re-downloaded into the ephemeral container layer.
_CONTAINER_HF_CACHE_PATH = "/root/.cache/huggingface"
# Torch hub cache mounted for parity with VGGT/DROID/Photo/MASt3R (GigaSLAM
# itself doesn't currently use torch.hub but upstream may add it).
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"
_CONTAINER_WORKDIR = "/gigaslam"


class GigaSLAMAlgorithm(SLAMAlgorithm):
    """GigaSLAM with dual conda+podman runtimes. Monocular KITTI only."""

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.gigaslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "GigaSLAM"
        self.conda_env = "gigaslam"
        self.docker_image = _PODMAN_IMAGE
        self._temp_config: Optional[Path] = None

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

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

        if self.container_runtime is None:
            # Conda path: rely on conda env existence; we don't probe binaries
            # here (matches DROID/VGGT/Photo/MASt3R conda-path preflight) —
            # the conda activate will fail loudly if missing.
            return

        # Podman path: need the runtime + image. Don't fail on missing image —
        # warn so the operator can build it.
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
                    self.gigaslam_path,
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

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.container_runtime is None:
            config_file = self._create_config(
                dataset_path=dataset_path,
                color_path_value=str(image_dir),
                save_dir_value=str(output_dir),
                output_dir=output_dir,
                ctx=ctx,
            )
            if not config_file:
                return None

            return ExecutionSpec(
                cmd=["gigaslam"],
                custom_runner=lambda _spec, config_file=config_file: self._run_gigaslam(config_file),
                log_prefix="GigaSLAM",
            )

        return self._build_container_execution_spec(
            ctx=ctx,
            dataset_path=dataset_path,
            image_dir=image_dir,
            output_dir=output_dir,
        )

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        dataset_path: Path,
        image_dir: Path,
        output_dir: Path,
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``gigaslam:latest``.

        The container ships the GigaSLAM source tree + the prebuilt
        ``sim3solve`` native CUDA extension + DBoW2 + DPRetrieval +
        diff-gaussian-rasterization + simple-knn.

        ``image_dir`` (host left-camera dir, e.g.
        ``datasets/kitti/sequences/04/image_2``) is bind-mounted at
        ``/dataset/kitti/sequences/<NN>/image_X`` so GigaSLAM's
        ``color_path.split("/")`` save-subdir derivation lands the same
        ``sequences_<NN>_image_X`` label as on the conda path.

        The wrapper generates a per-run YAML config inside ``output_dir``
        with ``Dataset.color_path`` rewritten to the container's view and
        ``Results.save_dir`` pointing at the container's ``/output``. The
        config file is bind-mounted at its absolute host path so the
        container reads the same file.
        """
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        hf_cache = Path.home() / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compose container-side paths that preserve the markers GigaSLAM's
        # save-subdir derivation reads (path[-3]/path[-2]/path[-1] in
        # slam.py:258 → ``sequences_<NN>_image_X``). We mount the SEQUENCE
        # ROOT (not just image_2/) so future configs or upstream changes
        # that read sibling files (calib.txt, times.txt, image_0/) keep
        # working transparently.
        sequence_root = image_dir.parent
        # The leaf image dir name on disk, e.g. "image_0" or "image_2".
        image_leaf = image_dir.name
        # Container-side sequence label: keep the same as the host sequence
        # dir name so the save-subdir derivation is stable.
        sequence_label = sequence_root.name
        container_sequence_root = (
            f"{_CONTAINER_DATASET_PARENT}/kitti/sequences/{sequence_label}"
        )
        container_image_dir = f"{container_sequence_root}/{image_leaf}"

        # Generate the per-run config with container-side paths so the file
        # the container reads contains values relative to the container's
        # filesystem (NOT the host's).
        config_file = self._create_config(
            dataset_path=dataset_path,
            color_path_value=container_image_dir,
            save_dir_value=_CONTAINER_OUTPUT_PATH,
            output_dir=output_dir,
            ctx=ctx,
        )
        if config_file is None:
            raise RuntimeError("GigaSLAM podman path: failed to create per-run config")
        host_config = config_file.resolve()
        # ``output_dir`` is bind-mounted at ``/output``; the per-run config
        # lives at ``output_dir/gigaslam_config.yaml``, so its container
        # path is deterministic.
        container_config = f"{_CONTAINER_OUTPUT_PATH}/{host_config.name}"

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            # GigaSLAM spawns mp.Queue + mp.Process workers under
            # torch.multiprocessing.set_start_method("spawn") (slam.py:8,
            # 122-126, 162, 229). These queues back IPC tensors in
            # /dev/shm. Podman's default 64 MB /dev/shm can SIGBUS the
            # backend right after spawn with no Python traceback. 8 GB
            # matches the per-keyframe Gaussian/depth tensor working set
            # plus IPC headroom (parity with DROID/Photo/MASt3R).
            "--shm-size=8g",
            # Mount the SEQUENCE ROOT read-only so GigaSLAM sees
            # /dataset/kitti/sequences/<NN>/image_X (and any sibling files
            # like calib.txt/times.txt) at the expected layout.
            "-v", f"{sequence_root.resolve()}:{container_sequence_root}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            # HuggingFace cache for UniDepthV2 weights (~1.3 GB for vitl14).
            "-v", f"{hf_cache.resolve()}:{_CONTAINER_HF_CACHE_PATH}",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
        ]

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # GigaSLAM always needs CUDA (UniDepthV2 + Gaussian Splatting +
        # sim3solve); ensure the GPU is always attached. HAMi adds extra
        # env/mounts on top.
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
        # harness is not active or when GigaSLAM's frame iterator hasn't
        # been wrapped — kept for parity with other foundation-model
        # wrappers so future deadline wiring is a one-line change).
        from ..runtime_stress.podman_injection import (
            apply_entrypoint_override,
            apply_realtime_to_podman_cmd,
        )
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)
        # The deadline-iterator hook lives in the frontend sub-module (the
        # frame loop is in utils/slam_frontend.py, not slam.py), so override
        # that file over the image's baked-in copy.
        apply_entrypoint_override(
            container_cmd,
            host_path=Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "GigaSLAM" / "utils" / "slam_frontend.py",
            container_path=f"{_CONTAINER_WORKDIR}/utils/slam_frontend.py",
        )

        # When HAMi is active, pre-create + bind-mount the per-container
        # shared cache file so GpuHamiController.apply() can mutate the
        # cap mid-run by writing directly to the mmap'd region.
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

        # Run GigaSLAM's slam.py entrypoint. The ``--eval`` flag enables
        # eval_rendering which is what writes ``poses_est.txt`` (per the
        # in-tree conda runner). Force PYTHONUNBUFFERED so the streaming
        # output reaches the SAL logs in real time.
        main_cmd = (
            f"cd {_CONTAINER_WORKDIR} && "
            f"PYTHONUNBUFFERED=1 python -u slam.py "
            f"--config {container_config} "
            f"--eval"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info(
            "  Executing GigaSLAM via %s container: %s",
            self.container_runtime, container_name,
        )

        io_target_paths = [str(sequence_root.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="GigaSLAM",
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
        """Build a unique podman-compliant container name for one GigaSLAM run."""
        raw_name = f"gigaslam-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

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
        color_path_value: str,
        save_dir_value: str,
        output_dir: Path,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Generate a per-run GigaSLAM config inheriting from the chosen base.

        The generated config lives in ``output_dir/gigaslam_config.yaml`` so
        it's reachable from inside the container (which bind-mounts
        ``output_dir`` at ``/output``).

        Args:
            dataset_path: Original dataset path (unused; kept for parity).
            color_path_value: Path string to write into ``Dataset.color_path``.
                For conda this is the host left-camera dir; for podman it is
                the container-side path.
            save_dir_value: Path string to write into ``Results.save_dir``.
                For conda this is the host ``output_dir``; for podman it is
                ``/output``.
            output_dir: Host output dir (where the generated YAML is written).
            ctx: Runtime context (used to look up the base config name).
        """
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

            config_text = re.sub(
                r'color_path:.*',
                f'color_path: "{color_path_value}"',
                config_text
            )

            if 'Results:' not in config_text:
                config_text = re.sub(
                    r'(inherit_from:.*\n)',
                    f'\\1\nResults:\n  save_dir: "{save_dir_value}"\n',
                    config_text
                )
            else:
                config_text = re.sub(
                    r'save_dir:.*',
                    f'save_dir: "{save_dir_value}"',
                    config_text
                )

            # Write modified config. Lives in output_dir so the container
            # can read it via the /output bind-mount.
            temp_config = output_dir / "gigaslam_config.yaml"
            with open(temp_config, 'w') as f:
                f.write(config_text)

            self._temp_config = temp_config
            logger.info(f"  Created config: {temp_config}")
            return temp_config

        except Exception as e:
            raise RuntimeError(f"Failed to create GigaSLAM config: {e}") from e

    def _run_gigaslam(self, config_file: Path) -> bool:
        """Execute GigaSLAM on the host conda env."""
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
        """Find raw GigaSLAM trajectory output.

        Same code path for conda and podman: GigaSLAM writes
        ``poses_est.txt`` under
        ``<save_dir>/<save-subdir>/<timestamp>{-LC|-No-LC}/poses_est.txt``
        (slam.py:259-262). The container's save_dir is bind-mounted to the
        host's ``output_dir`` so the file lands at the same relative path.
        """
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
        """Convert GigaSLAM trajectory to TUM format.

        Same code path for conda and podman: the trajectory layout on disk
        is identical in both runtimes (the container writes through the
        bind-mounted ``/output``).
        """
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
