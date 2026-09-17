"""MASt3R-SLAM algorithm implementation.

MASt3R-SLAM: Real-Time Dense SLAM with 3D Reconstruction Priors
(CVPR 2025). Foundation-model monocular SLAM sibling to VGGT-SLAM.

Two execution runtimes are supported, mirroring VGGT-SLAM, DROID-SLAM,
and Photo-SLAM:

* ``container_runtime=None`` (default): host conda environment
  ``mast3r-slam`` at ``deps/slam-algorithms/MASt3R-SLAM`` invoked via
  ``python main.py --dataset ... --config ...``.
* ``container_runtime="podman"``: ``mast3r-slam:latest`` Podman image.
  This path is the one runtime-stress scenarios use to exercise HAMi
  GPU caps against MASt3R-SLAM's foundation-model + ``mast3r_backends``
  + ``lietorch`` CUDA hot path.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


_PODMAN_IMAGE = "mast3r-slam:latest"
_CONTAINER_DATASET_PARENT = "/dataset"
# MASt3R-SLAM's load_dataset() picks the dataset class by string-matching
# 'tum' / 'euroc' / 'freiburg{N}' segments in the path. Mounting the
# staged TUM root at /dataset (no markers) would silently route to the
# generic RGBFiles fallback, which scans the directory for images and
# never reads rgb.txt -- leaving rgb_files empty. Build the container
# path so it carries both 'tum' and 'freiburg{N}' segments.
# MASt3R-SLAM writes the trajectory to ``logs/<dataset_stem>.txt`` at its
# working directory. Bind the host output dir at the container's logs
# path so the trajectory lands directly in the host output dir.
_CONTAINER_OUTPUT_PATH = "/mast3r-slam/logs"
_CONTAINER_CHECKPOINTS_PATH = "/mast3r-slam/checkpoints"
_CONTAINER_CONFIG_PATH = "/mast3r-slam/config"
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"
_CONTAINER_WORKDIR = "/mast3r-slam"


class MASt3RSLAMAlgorithm(SLAMAlgorithm):
    """MASt3R-SLAM with dual conda+podman runtimes (mono TUM only today)."""

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.mast3r_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "MASt3R-SLAM"
        self.conda_env = "mast3r-slam"
        self.docker_image = _PODMAN_IMAGE

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

    @property
    def name(self) -> str:
        return "mast3rslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "tum": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        # eval_calib requires calibration which most perturbed datasets don't have
        return "base"

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return self.mast3r_path / "config" / f"{ctx.internal_config_name}.yaml"

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate MASt3R-SLAM runtime dependencies."""
        if not self.mast3r_path.exists():
            raise RuntimeError(f"MASt3R-SLAM not found at {self.mast3r_path}")

        config_file = self._resolve_config_path(ctx)

        if config_file is None:
            raise RuntimeError("MASt3R-SLAM external config path is not resolved in runtime context")

        if not config_file.exists():
            raise RuntimeError(f"MASt3R-SLAM config file not found: {config_file}")

        if request.dataset_type.lower() == "tum":
            self._resolve_freiburg_id_from_sequence(ctx.sequence_name)

        if self.container_runtime is None:
            # Conda path needs the env to exist plus the in-repo checkpoints.
            # We don't probe conda binaries here (matches DROID/VGGT/Photo
            # conda-path preflight) — the conda activate will fail loudly if
            # missing. Just check that the upstream checkpoint files MASt3R-SLAM
            # needs at runtime are in tree.
            self._require_mast3r_checkpoints()
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
                    self.mast3r_path,
                    self.docker_image,
                )
        except Exception as exc:
            logger.info(
                "Skipping podman image existence check: %s", exc,
            )

        # Podman path still depends on the host's MASt3R-SLAM checkpoints
        # directory because the image bind-mounts it read-only at runtime
        # (the foundation-model weights are 2.7 GB; bundling them into the
        # image would balloon it from ~12 GB to ~15 GB and slow `podman pull`
        # on every reuse).
        self._require_mast3r_checkpoints()

    def _require_mast3r_checkpoints(self) -> None:
        """Validate that the MASt3R-SLAM foundation-model weights are in tree."""
        checkpoints = self.mast3r_path / "checkpoints"
        primary = checkpoints / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        if not primary.exists():
            raise RuntimeError(
                f"MASt3R-SLAM checkpoint not found at {primary}. "
                f"Run: cd {self.mast3r_path} && ./install_all.sh "
                "(or download manually from https://download.europe.naverlabs.com/ComputerVision/MASt3R)."
            )

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Prepare dataset layout expected by MASt3R-SLAM."""
        return self._prepare_dataset(request, ctx.sequence_name)

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Clean temporary symlinked TUM staging directories."""
        self._cleanup_temp_tum_link()

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by MASt3R-SLAM execution."""
        prepared_path = ctx.effective_dataset_path
        config_file = self._resolve_config_path(ctx)
        if not prepared_path or config_file is None:
            return None
        return {
            "output_dir": request.output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "prepared_path": prepared_path,
            "config_file": config_file,
            "dataset_type": request.dataset_type,
            "sequence_name": ctx.sequence_name,
            "log_basenames": self._expected_log_basenames(request.dataset_type, ctx.sequence_name),
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        inputs = ctx.execution_inputs
        prepared_path = inputs["prepared_path"]
        config_file = inputs["config_file"]
        output_dir = inputs["output_dir"]
        log_basenames = inputs["log_basenames"]

        if self.container_runtime is None:
            return ExecutionSpec(
                cmd=["mast3rslam"],
                custom_runner=lambda _spec, prepared_path=prepared_path, config_file=config_file, output_dir=output_dir, log_basenames=log_basenames: self._run_mast3rslam(
                    prepared_path,
                    config_file,
                    output_dir,
                    log_basenames,
                ),
                log_prefix="MASt3R-SLAM",
            )

        return self._build_container_execution_spec(
            ctx=ctx,
            prepared_path=prepared_path,
            config_file=config_file,
            output_dir=output_dir,
        )

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        prepared_path: Path,
        config_file: Path,
        output_dir: Path,
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``mast3r-slam:latest``.

        The container ships the MASt3R-SLAM source tree + the prebuilt
        ``mast3r_slam_backends`` native CUDA extension + lietorch + the
        in3d/mast3r thirdparty packages.

        ``prepared_path`` (the host's staged TUM root from
        ``_prepare_dataset``) is bind-mounted under ``_CONTAINER_DATASET_PARENT``
        at a path whose components include 'tum' and 'freiburg{N}' markers
        that MASt3R-SLAM's ``load_dataset`` string-matches on.
        Because that staged dir uses an absolute-path symlink for ``rgb/``
        pointing into the real perturbed camera dir, we also bind-mount
        that target at its same host path inside the container so the
        symlink resolves.

        The trajectory file is written to ``/mast3r-slam/logs/<stem>.txt``
        by ``mast3r_slam.evaluate.save_traj``; we bind-mount the host
        output dir at that exact path so the trajectory lands directly on
        the host with no post-exit copy.
        """
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoints (2.9 GB) live in the host repo at
        # deps/slam-algorithms/MASt3R-SLAM/checkpoints. The Dockerfile
        # intentionally does NOT COPY them; bind-mount read-only at runtime.
        host_checkpoints = self.mast3r_path / "checkpoints"

        # MASt3R-SLAM reads ``config/<name>.yaml`` relative to the working
        # dir. Map the host's resolved config file into the container's
        # config tree by relative path under mast3r_path. If the file lives
        # outside mast3r_path (external config), fall back to bind-mounting
        # the file's absolute host path.
        host_config = config_file.resolve()
        container_config = self._translate_config_path_to_container(host_config)

        # Build a container dataset path that carries the dataset-type and
        # freiburg-variant markers MASt3R-SLAM's ``load_dataset`` matches on.
        freiburg_id = self._resolve_freiburg_id_from_sequence(ctx.sequence_name)
        container_dataset_path = (
            f"{_CONTAINER_DATASET_PARENT}/tum/rgbd_dataset_{freiburg_id}_data"
        )

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            # MASt3R-SLAM allocates SharedKeyframes/SharedStates over
            # multiprocessing.Manager, which backs IPC tensors in /dev/shm.
            # Podman's default 64 MB /dev/shm is too small and SharedStates
            # init dies with SIGBUS (exit 135) right after config print, no
            # Python traceback. 8 GB matches DROID-SLAM/Photo-SLAM scale.
            "--shm-size=8g",
            "-v", f"{prepared_path.resolve()}:{container_dataset_path}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            "-v", f"{host_checkpoints.resolve()}:{_CONTAINER_CHECKPOINTS_PATH}:ro",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
        ]

        # The staged TUM root's rgb/ entry is a symlink to the perturbed
        # camera dir (absolute host path); bind-mount that target so the
        # symlink resolves inside the container.
        for host_path in self._collect_symlink_target_mounts(prepared_path):
            container_cmd.extend(["-v", f"{host_path}:{host_path}:ro"])

        # External configs land at their absolute host path; bind-mount
        # the file so the container can read it.
        if container_config == str(host_config):
            container_cmd.extend(
                ["-v", f"{host_config}:{host_config}:ro"]
            )

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # MASt3R-SLAM always needs CUDA (foundation model + lietorch +
        # mast3r_backends); ensure the GPU is always attached. HAMi adds
        # extra env/mounts on top.
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
        from ..runtime_stress.podman_injection import (
            apply_entrypoint_override,
            apply_realtime_to_podman_cmd,
        )
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)

        # Override the container's baked-in snapshot of main.py with the
        # host's current copy so the deadline-harness hooks (and any other
        # host-side edit) reach the SLAM without a 15-25 min image rebuild.
        apply_entrypoint_override(
            container_cmd,
            host_path=Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "MASt3R-SLAM" / "main.py",
            container_path=f"{_CONTAINER_WORKDIR}/main.py",
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

        # MASt3R-SLAM's main.py uses torch.multiprocessing.set_start_method
        # ("spawn") which requires the in3d/mast3r thirdparty packages on
        # PYTHONPATH. The Dockerfile pip installs them so they're on path
        # already, but mirror the host wrapper's PYTHONPATH composition to
        # be defensive against image-side path skew.
        main_cmd = (
            f"cd {_CONTAINER_WORKDIR} && "
            f"PYTHONPATH=\"{_CONTAINER_WORKDIR}/thirdparty/in3d:"
            f"{_CONTAINER_WORKDIR}/thirdparty/mast3r:"
            f"{_CONTAINER_WORKDIR}:${{PYTHONPATH:-}}\" "
            f"python main.py "
            f"--dataset {container_dataset_path} "
            f"--config {container_config} "
            f"--no-viz"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info(
            "  Executing MASt3R-SLAM via %s container: %s",
            self.container_runtime, container_name,
        )

        io_target_paths = [str(prepared_path.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="MASt3R-SLAM",
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
        """Build a unique podman-compliant container name for one MASt3R-SLAM run."""
        raw_name = f"mast3rslam-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

    def _translate_config_path_to_container(self, host_config_path: Path) -> str:
        """Map a host-side MASt3R-SLAM config path to its container path.

        The ``config/`` tree is COPYed into the image verbatim, so the
        relative layout matches. External configs (anywhere on the host)
        fall back to using their absolute host path inside the container,
        which requires the operator's bind-mount to make them visible;
        we log a warning so a misconfigured external config doesn't fail
        silently.
        """
        try:
            relative = host_config_path.resolve().relative_to(self.mast3r_path.resolve())
        except ValueError:
            logger.warning(
                "  External MASt3R-SLAM config %s is outside of %s; the "
                "container path will not resolve unless the file is "
                "bind-mounted. The wrapper bind-mounts it at its absolute "
                "host path so this works as long as the file is readable.",
                host_config_path, self.mast3r_path,
            )
            return str(host_config_path.resolve())
        return f"{_CONTAINER_WORKDIR}/{relative.as_posix()}"

    def _collect_symlink_target_mounts(self, dataset_root: Path) -> List[str]:
        """Return absolute host paths to bind-mount so absolute-path
        symlinks under ``dataset_root`` resolve inside the container.

        ``_prepare_dataset`` builds the staged TUM root by writing an
        absolute symlink ``rgb -> /abs/path/to/perturbed_rgb``. Inside the
        container, that symlink points to a host-absolute path the
        container can't see unless we bind-mount its target. This helper
        walks the staged tree once and returns the set of target paths the
        container needs.
        """
        if not dataset_root.exists():
            return []

        seen: List[str] = []
        seen_set: set[str] = set()

        def _maybe_record(entry: Path) -> None:
            if not entry.is_symlink():
                return
            try:
                target = Path(os.readlink(entry))
            except OSError:
                return
            if not target.is_absolute():
                # Relative-path symlinks resolve naturally inside the
                # container because the host_root is already mounted.
                return
            target_str = str(target.resolve())
            if target_str in seen_set:
                return
            if not Path(target_str).exists():
                return
            seen.append(target_str)
            seen_set.add(target_str)

        for root, dirs, files in os.walk(str(dataset_root), followlinks=False):
            root_path = Path(root)
            for d in dirs:
                _maybe_record(root_path / d)
            for f in files:
                _maybe_record(root_path / f)
        # Top-level symlinks (siblings of dataset_root) aren't covered by
        # os.walk's traversal of dataset_root itself; cover those too.
        for entry in dataset_root.iterdir():
            _maybe_record(entry)
        return seen

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]
        config_file = inputs["config_file"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  MASt3R-SLAM only supports monocular mode. Stereo flag ignored.")

        logger.info(f"  Using config: {config_file}")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run MASt3R-SLAM: {e}")
            return False

    def _prepare_dataset(
        self,
        request: SLAMRunRequest,
        sequence_name: str,
    ) -> Path:
        """Prepare dataset for MASt3R-SLAM.

        MASt3R-SLAM's TUM dataloader requires:
        - dataset path includes a 'tum' directory segment
        - dataset path includes 'freiburg{1,2,3}' marker
        - dataset root contains rgb.txt with entries like: "<timestamp> rgb/<filename>"
        - dataset root contains rgb/ images

        We enforce a deterministic, contract-driven staging path so we don't rely on
        source path probing or mutate input datasets.
        """
        logger.info("  Preparing dataset for MASt3R-SLAM...")
        dataset_type = request.dataset_type.lower()
        if dataset_type != "tum":
            raise ValueError(
                f"MASt3R-SLAM supports only TUM monocular datasets, got dataset_type='{request.dataset_type}'."
            )

        freiburg_id = self._resolve_freiburg_id_from_sequence(sequence_name)
        left_image_dir = self._require_left_camera_path(request)
        image_files = self._collect_sorted_image_files(left_image_dir)
        timestamps_by_frame = self._require_timestamps_by_frame(request)

        expected_frame_count = len(image_files)
        if len(timestamps_by_frame) != expected_frame_count:
            raise ValueError(
                "MASt3R-SLAM staging requires timestamp and image counts to match: "
                f"timestamps={len(timestamps_by_frame)}, images={expected_frame_count}."
            )

        stage_root = Path(tempfile.mkdtemp(prefix="tum_dataset_"))
        staged_dataset_path = (
            stage_root
            / "tum"
            / f"rgbd_dataset_{freiburg_id}_{self._sanitize_sequence_label(sequence_name)}"
        )
        staged_dataset_path.mkdir(parents=True, exist_ok=True)

        rgb_link = staged_dataset_path / "rgb"
        os.symlink(
            str(left_image_dir.resolve()),
            str(rgb_link),
            target_is_directory=True,
        )

        self._write_staged_rgb_txt(
            staged_dataset_path=staged_dataset_path,
            image_files=image_files,
            timestamps_by_frame=timestamps_by_frame,
        )

        logger.info("  Created MASt3R staged TUM dataset: %s", staged_dataset_path)
        self._temp_tum_link = staged_dataset_path
        return staged_dataset_path

    def _resolve_freiburg_id_from_sequence(self, sequence_name: str) -> str:
        """Resolve Freiburg variant from dataset.sequence for TUM runs.

        Accepted markers:
        - freiburg1 / fr1
        - freiburg2 / fr2
        - freiburg3 / fr3

        Raises:
            ValueError: if the sequence is missing or ambiguous.
        """
        sequence = (sequence_name or "").strip().lower()
        if not sequence:
            raise ValueError(
                "MASt3R-SLAM TUM calibration requires dataset.sequence to include Freiburg variant "
                "(freiburg1/freiburg2/freiburg3 or fr1/fr2/fr3)."
            )

        tokens = [t for t in re.split(r"[^a-z0-9]+", sequence) if t]
        variants: List[str] = []
        if "freiburg1" in tokens or "fr1" in tokens:
            variants.append("freiburg1")
        if "freiburg2" in tokens or "fr2" in tokens:
            variants.append("freiburg2")
        if "freiburg3" in tokens or "fr3" in tokens:
            variants.append("freiburg3")

        if len(variants) == 1:
            return variants[0]

        if not variants:
            raise ValueError(
                f"MASt3R-SLAM could not resolve Freiburg variant from dataset.sequence='{sequence_name}'. "
                "Use one of: freiburg1/freiburg2/freiburg3 (or fr1/fr2/fr3)."
            )

        raise ValueError(
            f"MASt3R-SLAM found ambiguous Freiburg markers in dataset.sequence='{sequence_name}': {variants}. "
            "Provide exactly one Freiburg variant (freiburg1/freiburg2/freiburg3)."
        )

    def _require_left_camera_path(self, request: SLAMRunRequest) -> Path:
        """Resolve and validate left camera path from runtime request extras."""
        camera_paths = request.extras.get("camera_paths")
        if not isinstance(camera_paths, dict):
            raise ValueError(
                "MASt3R-SLAM requires request.extras['camera_paths'] as a role->path mapping."
            )

        left_path_value = camera_paths.get("left")
        if not isinstance(left_path_value, str) or not left_path_value.strip():
            raise ValueError(
                "MASt3R-SLAM requires request.extras['camera_paths']['left']."
            )

        left_path = Path(left_path_value).resolve()
        if not left_path.exists() or not left_path.is_dir():
            raise ValueError(
                f"MASt3R-SLAM left camera directory does not exist: {left_path}"
            )
        return left_path

    def _collect_sorted_image_files(self, image_dir: Path) -> List[Path]:
        """Collect sorted image files from a camera directory."""
        image_files = sorted(
            list(image_dir.glob("*.png"))
            + list(image_dir.glob("*.jpg"))
            + list(image_dir.glob("*.jpeg"))
        )
        if not image_files:
            raise ValueError(
                f"MASt3R-SLAM found no PNG/JPG/JPEG images under left camera directory: {image_dir}"
            )
        return image_files

    def _require_timestamps_by_frame(self, request: SLAMRunRequest) -> Dict[int, Union[int, float]]:
        """Resolve and validate frame-indexed timestamps from request extras."""
        raw_timestamps = request.extras.get("timestamps_by_frame")
        if not isinstance(raw_timestamps, dict) or not raw_timestamps:
            raise ValueError(
                "MASt3R-SLAM requires request.extras['timestamps_by_frame']."
            )

        normalized: Dict[int, Union[int, float]] = {}
        for frame_idx, timestamp in raw_timestamps.items():
            if not isinstance(frame_idx, int):
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamps_by_frame key {frame_idx!r}; expected int frame index."
                )
            if frame_idx < 0:
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamps_by_frame key {frame_idx}; expected non-negative frame index."
                )
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"MASt3R-SLAM invalid timestamp value for frame {frame_idx}: {timestamp!r}"
                )
            normalized[frame_idx] = timestamp

        expected_indices = set(range(len(normalized)))
        actual_indices = set(normalized.keys())
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            extra = sorted(actual_indices - expected_indices)
            raise ValueError(
                "MASt3R-SLAM timestamps_by_frame must use contiguous frame indices 0..N-1 "
                f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})."
            )

        previous: Optional[float] = None
        for frame_idx in range(len(normalized)):
            current = float(normalized[frame_idx])
            if previous is not None and current <= previous:
                raise ValueError(
                    "MASt3R-SLAM timestamps_by_frame must be strictly increasing by frame index: "
                    f"frame {frame_idx - 1}={previous}, frame {frame_idx}={current}."
                )
            previous = current

        return normalized

    def _write_staged_rgb_txt(
        self,
        staged_dataset_path: Path,
        image_files: List[Path],
        timestamps_by_frame: Dict[int, Union[int, float]],
    ) -> None:
        """Write MASt3R-compatible TUM rgb.txt in staged dataset root."""
        rgb_txt = staged_dataset_path / "rgb.txt"
        with open(rgb_txt, "w", encoding="utf-8") as file_handle:
            for frame_idx, image_path in enumerate(image_files):
                timestamp = timestamps_by_frame[frame_idx]
                file_handle.write(f"{timestamp} rgb/{image_path.name}\n")

    @staticmethod
    def _sanitize_sequence_label(sequence_name: str) -> str:
        """Create a filesystem-safe label from a canonical sequence name."""
        cleaned = "".join(
            ch if ch.isalnum() or ch in {"-", "_", "."} else "_"
            for ch in sequence_name.strip()
        )
        return cleaned or "sequence"

    def _expected_log_basenames(self, dataset_type: str, sequence_name: str) -> List[str]:
        """Return deterministic MASt3R log basenames for this run."""
        safe_sequence = self._sanitize_sequence_label(sequence_name)
        basenames = [safe_sequence]
        if dataset_type == "tum":
            freiburg_id = self._resolve_freiburg_id_from_sequence(sequence_name)
            basenames.insert(0, f"rgbd_dataset_{freiburg_id}_{safe_sequence}")
        return basenames

    def _run_mast3rslam(
        self,
        dataset_path: Path,
        config_file: Path,
        output_dir: Path,
        log_basenames: List[str],
    ) -> bool:
        """Execute MASt3R-SLAM via host conda environment."""
        logger.info("  Executing MASt3R-SLAM...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        # MASt3R-SLAM saves to logs/{dataset_name}.txt
        cmd = (
            f"cd {self.mast3r_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"PYTHONPATH=\"{self.mast3r_path}/thirdparty/in3d:{self.mast3r_path}/thirdparty/mast3r:{self.mast3r_path}:${{PYTHONPATH:-}}\" "
            f"python main.py --dataset {dataset_path} --config {config_file} --no-viz"
        )

        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
            )

            self._stream_process_output(process, "MASt3R-SLAM")

            self._wait_for_process(process, timeout_seconds=7200)

            if process.returncode != 0:
                logger.error(f"MASt3R-SLAM failed with return code {process.returncode}")
                return False

            logger.info("  MASt3R-SLAM completed successfully")

            traj_src = self._locate_host_logs_trajectory(log_basenames)
            if traj_src is None:
                return False

            traj_dst = output_dir / "CameraTrajectory.txt"
            shutil.copy2(traj_src, traj_dst)
            logger.info(f"  Trajectory copied to {traj_dst}")

            return True

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("MASt3R-SLAM timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run MASt3R-SLAM: {e}")
            return False

    def _locate_host_logs_trajectory(self, log_basenames: List[str]) -> Optional[Path]:
        """Locate the MASt3R-SLAM-written trajectory under ``logs/`` on the host.

        Conda path only — Podman writes directly into ``output_dir`` via the
        bind-mount, so this is not called there.
        """
        logs_dir = self.mast3r_path / "logs"
        for basename in log_basenames:
            candidate = logs_dir / f"{basename}.txt"
            if candidate.exists():
                logger.info(f"  Found trajectory: {candidate.name}")
                return candidate

        expected_files = [f"{basename}.txt" for basename in log_basenames]
        available_files = sorted(path.name for path in logs_dir.glob("*.txt"))
        logger.error(
            "  No matching MASt3R trajectory file found. "
            f"Expected one of: {expected_files}. "
            f"Available: {available_files if available_files else '[none]'}"
        )
        return None

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw MASt3R-SLAM trajectory output.

        Conda path: ``_run_mast3rslam`` already copies the trajectory to
        ``output_dir/CameraTrajectory.txt``.

        Podman path: the container writes ``<output_dir>/<basename>.txt``
        via the bind-mount (host ``output_dir`` → container
        ``/mast3r-slam/logs``). Locate that file by basename.
        """
        camera_traj = request.output_dir / "CameraTrajectory.txt"
        if camera_traj.exists():
            return camera_traj

        if self.container_runtime is None:
            return None

        inputs = ctx.execution_inputs or {}
        log_basenames = inputs.get("log_basenames") or self._expected_log_basenames(
            request.dataset_type, ctx.sequence_name
        )
        for basename in log_basenames:
            candidate = request.output_dir / f"{basename}.txt"
            if candidate.exists():
                return candidate

        # As a last resort, accept any *.txt in the output dir that matches
        # the freiburg pattern — useful when MASt3R upgrades and renames
        # the trajectory file.
        for txt_path in sorted(request.output_dir.glob("*.txt")):
            if "rgbd_dataset" in txt_path.name or "freiburg" in txt_path.name.lower():
                return txt_path
        return None

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """MASt3R-SLAM trajectory is already TUM-compatible.

        Conda path: ``raw_trajectory`` is ``output_dir/CameraTrajectory.txt``.
        Podman path: ``raw_trajectory`` is ``output_dir/<basename>.txt``;
        copy it to the canonical ``output_dir/CameraTrajectory.txt``.
        """
        canonical = request.output_dir / "CameraTrajectory.txt"
        if raw_trajectory.resolve() == canonical.resolve():
            return canonical
        try:
            shutil.copy2(raw_trajectory, canonical)
            logger.info(f"  Trajectory copied to {canonical}")
        except Exception as exc:
            logger.error(f"  Failed to copy trajectory to canonical path: {exc}")
            return None
        return canonical

    def cleanup(self) -> None:
        """Clean up temporary files."""
        self._cleanup_temp_tum_link()

    def _cleanup_temp_tum_link(self) -> None:
        """Remove temporary TUM symlink staging directory when present."""
        if hasattr(self, "_temp_tum_link") and self._temp_tum_link:
            try:
                temp_dir = self._temp_tum_link.parent.parent
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            self._temp_tum_link = None
