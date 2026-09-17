"""Photo-SLAM algorithm implementation.

Photo-SLAM: Real-time Simultaneous Localization and Photorealistic Mapping
for Monocular, Stereo, and RGB-D Cameras (CVPR 2024).

Two execution runtimes are supported, mirroring VGGT-SLAM and DROID-SLAM:

* ``container_runtime=None`` (default): host build at
  ``deps/slam-algorithms/Photo-SLAM`` invoked directly via the C++ binaries
  in ``bin/`` (``tum_mono``, ``tum_rgbd``, ``euroc_stereo``).
* ``container_runtime="podman"``: ``photoslam:latest`` Podman image. This
  path is the one runtime-stress scenarios use to exercise HAMi GPU caps
  against Photo-SLAM's Gaussian Splatting + LibTorch hot path.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


_PODMAN_IMAGE = "photoslam:latest"
_CONTAINER_DATASET_PATH = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
_CONTAINER_ASSOC_PATH = "/dataset_meta/associations.txt"
_CONTAINER_EUROC_TIMESTAMPS_PATH = "/dataset_meta/photoslam_timestamps.txt"
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"
_CONTAINER_WORKDIR = "/photo-slam"


class PhotoSLAMAlgorithm(SLAMAlgorithm):
    """Photo-SLAM algorithm via native C++ executables.

    Supports TUM (mono, rgbd) and EuRoC (stereo) datasets.
    Uses ORB-SLAM3 for tracking with Gaussian Splatting for mapping.
    """

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.photoslam_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "Photo-SLAM"
        self.docker_image = _PODMAN_IMAGE
        self._staged_dataset_dir: Optional[Path] = None

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

    @property
    def name(self) -> str:
        return "photoslam"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "tum": ["mono", "rgbd"],
            "euroc": ["stereo"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        """Map dataset sequence to Photo-SLAM config name."""
        if dataset_type.lower() == "tum":
            seq_lower = str(sequence).lower()
            if "freiburg1" in seq_lower or "fr1" in seq_lower:
                if "desk" in seq_lower:
                    return "tum_freiburg1_desk"
                return "tum_freiburg1_desk"  # Default for fr1
            elif "freiburg2" in seq_lower or "fr2" in seq_lower:
                if "xyz" in seq_lower:
                    return "tum_freiburg2_xyz"
                return "tum_freiburg2_xyz"  # Default for fr2
            elif "freiburg3" in seq_lower or "fr3" in seq_lower:
                if "office" in seq_lower or "household" in seq_lower:
                    return "tum_freiburg3_long_office_household"
                return "tum_freiburg3_long_office_household"  # Default for fr3
            # Generic TUM config
            return "tum_mono"
        elif dataset_type.lower() == "euroc":
            # All EuRoC sequences use the same config
            return "EuRoC"
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Photo-SLAM resolves multiple config files by mode/dataset at runtime."""
        return None

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate Photo-SLAM runtime dependencies."""
        if self.container_runtime is None:
            # Host build: need the source tree and the built C++ binaries.
            if not self.photoslam_path.exists():
                raise RuntimeError(f"Photo-SLAM not found at {self.photoslam_path}")

            bin_dir = self.photoslam_path / "bin"
            if not bin_dir.exists():
                raise RuntimeError(
                    f"Photo-SLAM is not built. Missing directory: {bin_dir}. "
                    f"Run: {self.photoslam_path}/install_all.sh"
                )
            return

        # Podman path: need the runtime + image. Don't fail on missing image —
        # warn so the operator can build it. The C++ binaries ship inside the
        # image, so the host's bin/ dir is not required.
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
                    "cd %s && ./build_image.sh "
                    "(stages the C++ deadline iterator from "
                    "src/runtime_stress, then runs podman build).",
                    self.docker_image,
                    self.photoslam_path,
                )
        except Exception as exc:
            logger.info(
                "Skipping podman image existence check: %s", exc,
            )

    def _stage_dataset(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Stage a strict Photo-SLAM runtime dataset layout."""
        dataset_type = request.dataset_type.lower()
        if dataset_type not in {"tum", "euroc"}:
            raise RuntimeError(f"Photo-SLAM does not support dataset type '{request.dataset_type}'.")

        staged_root = Path(tempfile.mkdtemp(prefix="photoslam_stage_"))
        self._staged_dataset_dir = staged_root
        ctx.staging_artifacts["photoslam_stage_root"] = staged_root

        try:
            if dataset_type == "tum":
                self._prepare_tum_stage(request, ctx, staged_root)
            else:
                self._prepare_euroc_stage(request, ctx, staged_root)
        except Exception:
            shutil.rmtree(staged_root, ignore_errors=True)
            self._staged_dataset_dir = None
            ctx.staging_artifacts.pop("photoslam_stage_root", None)
            ctx.staging_artifacts.pop("photoslam_association_file", None)
            ctx.staging_artifacts.pop("photoslam_timestamps_file", None)
            raise

        return staged_root

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Clean up Photo-SLAM staged dataset directory."""
        staged_root = ctx.staging_artifacts.pop("photoslam_stage_root", None)
        if isinstance(staged_root, Path) and staged_root.exists():
            shutil.rmtree(staged_root, ignore_errors=True)
        self._staged_dataset_dir = None
        ctx.staging_artifacts.pop("photoslam_association_file", None)
        ctx.staging_artifacts.pop("photoslam_timestamps_file", None)

    def _prepare_tum_stage(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
        staged_root: Path,
    ) -> None:
        """Prepare staged TUM layout and enforce strict association requirements."""
        left_image_dir = self._require_camera_path(request, "left")

        self._symlink_tree_entries(request.dataset_path, staged_root)

        rgb_link = staged_root / "rgb"
        self._replace_with_symlink(rgb_link, left_image_dir)

        depth_dir = request.dataset_path / "depth"
        use_rgbd = depth_dir.exists() and depth_dir.is_dir() and any(depth_dir.iterdir())
        if not use_rgbd:
            return

        assoc_file = self._find_existing_tum_association_file(request.dataset_path)
        if assoc_file is None:
            raise RuntimeError(
                "Photo-SLAM TUM RGB-D mode requires an existing association file in dataset root "
                "(expected one of: associations.txt, association.txt, associate.txt, assoc.txt)."
            )

        staged_assoc = staged_root / assoc_file.name
        if not staged_assoc.exists():
            raise RuntimeError(
                f"Photo-SLAM staged association file is missing: {staged_assoc}."
            )
        ctx.staging_artifacts["photoslam_association_file"] = staged_assoc

    def _prepare_euroc_stage(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
        staged_root: Path,
    ) -> None:
        """Prepare staged EuRoC layout with strict camera/timestamp contracts."""
        left_images_dir = self._require_camera_path(request, "left")
        right_images_dir = self._require_camera_path(request, "right")

        timestamps_by_frame = self._require_euroc_timestamps_by_frame(request)
        left_images = self._list_camera_images(left_images_dir)
        right_images = self._list_camera_images(right_images_dir)

        if len(left_images) != len(right_images):
            raise RuntimeError(
                "Photo-SLAM EuRoC staging requires synchronized stereo image counts "
                f"(left={len(left_images)}, right={len(right_images)})."
            )

        if len(left_images) != len(timestamps_by_frame):
            raise RuntimeError(
                "Photo-SLAM EuRoC timestamps must match stereo image count "
                f"(timestamps={len(timestamps_by_frame)}, images={len(left_images)})."
            )

        self._symlink_tree_entries(request.dataset_path, staged_root, exclude_names={"mav0"})

        source_mav0 = request.dataset_path / "mav0"
        staged_mav0 = staged_root / "mav0"
        staged_mav0.mkdir(parents=True, exist_ok=True)
        if source_mav0.exists() and source_mav0.is_dir():
            self._symlink_tree_entries(source_mav0, staged_mav0, exclude_names={"cam0", "cam1"})

        staged_cam0 = staged_mav0 / "cam0"
        staged_cam1 = staged_mav0 / "cam1"
        staged_cam0.mkdir(parents=True, exist_ok=True)
        staged_cam1.mkdir(parents=True, exist_ok=True)

        source_cam0 = source_mav0 / "cam0"
        source_cam1 = source_mav0 / "cam1"
        if source_cam0.exists() and source_cam0.is_dir():
            self._symlink_tree_entries(source_cam0, staged_cam0, exclude_names={"data"})
        if source_cam1.exists() and source_cam1.is_dir():
            self._symlink_tree_entries(source_cam1, staged_cam1, exclude_names={"data"})

        self._replace_with_symlink(staged_cam0 / "data", left_images_dir)
        self._replace_with_symlink(staged_cam1 / "data", right_images_dir)

        timestamps_file = staged_root / "photoslam_timestamps.txt"
        self._write_euroc_timestamps_file(timestamps_by_frame, timestamps_file)
        ctx.staging_artifacts["photoslam_timestamps_file"] = timestamps_file

    def _symlink_tree_entries(
        self,
        source_dir: Path,
        target_dir: Path,
        exclude_names: Optional[set[str]] = None,
    ) -> None:
        """Symlink immediate children from source_dir into target_dir."""
        excluded = exclude_names or set()
        target_dir.mkdir(parents=True, exist_ok=True)

        for entry in sorted(source_dir.iterdir(), key=lambda p: p.name):
            if entry.name in excluded:
                continue
            self._replace_with_symlink(target_dir / entry.name, entry)

    def _replace_with_symlink(self, link_path: Path, target_path: Path) -> None:
        """Replace link_path with a symlink to target_path."""
        if link_path.is_symlink():
            link_path.unlink()
        elif link_path.exists():
            if link_path.is_dir():
                shutil.rmtree(link_path)
            else:
                link_path.unlink()
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(target_path.resolve(), target_is_directory=target_path.is_dir())

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by Photo-SLAM execution."""
        if ctx.effective_dataset_path is None:
            return None

        slam_config = (
            str(ctx.resolved_config_path)
            if ctx.config_is_external and ctx.resolved_config_path is not None
            else str(ctx.internal_config_name or request.slam_config)
        )
        return {
            "dataset_path": ctx.effective_dataset_path,
            "slam_config": slam_config,
            "output_dir": request.output_dir,
            "dataset_type": request.dataset_type,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "is_external": ctx.config_is_external,
            "staged_association_file": ctx.staging_artifacts.get("photoslam_association_file"),
            "staged_timestamps_file": ctx.staging_artifacts.get("photoslam_timestamps_file"),
        }

    def _build_execution_spec(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[ExecutionSpec]:
        """Build execution specification for Photo-SLAM backend."""
        inputs = ctx.execution_inputs
        dataset_path = inputs["dataset_path"]
        slam_config = inputs["slam_config"]
        output_dir = inputs["output_dir"]
        dataset_type = inputs["dataset_type"]
        is_stereo = inputs["is_stereo"]
        is_external = inputs["is_external"]
        staged_association_file = inputs["staged_association_file"]
        staged_timestamps_file = inputs["staged_timestamps_file"]
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve the per-mode binary + configs first. These selections
        # are the same regardless of conda vs podman; only the absolute
        # paths embedded into the final command differ.
        if dataset_type.lower() == "tum":
            if is_stereo:
                logger.error("  Photo-SLAM TUM does not support stereo mode.")
                return None

            depth_path = dataset_path / "depth"
            use_rgbd = depth_path.exists() and depth_path.is_dir() and any(depth_path.iterdir())

            if use_rgbd:
                bin_name = "tum_rgbd"
                mode = "RGB-D"
            else:
                bin_name = "tum_mono"
                mode = "Monocular"

            orb_config = self._get_orb_config(slam_config, mode, "TUM", is_external)
            gaussian_config = self._get_gaussian_config(slam_config, mode, "TUM")

            if not orb_config or not gaussian_config:
                return None

            if use_rgbd:
                if not isinstance(staged_association_file, Path) or not staged_association_file.exists():
                    logger.error(
                        "  Photo-SLAM TUM RGB-D mode requires a staged association file; "
                        "dataset staging did not provide one."
                    )
                    return None

        elif dataset_type.lower() == "euroc":
            if not is_stereo:
                logger.error("  Photo-SLAM EuRoC requires stereo mode.")
                return None

            bin_name = "euroc_stereo"
            mode = "Stereo"

            orb_config = self._get_orb_config(slam_config, mode, "EuRoC", is_external)
            gaussian_config = self._get_gaussian_config(slam_config, mode, "EuRoC")

            if not orb_config or not gaussian_config:
                return None

            if not isinstance(staged_timestamps_file, Path) or not staged_timestamps_file.exists():
                logger.error(
                    "  Photo-SLAM EuRoC mode requires a staged timestamps file; dataset staging did not provide one."
                )
                return None
        else:
            logger.error(f"Unsupported dataset type: {dataset_type}")
            return None

        if self.container_runtime is None:
            return self._build_host_execution_spec(
                bin_name=bin_name,
                orb_config=orb_config,
                gaussian_config=gaussian_config,
                dataset_path=dataset_path,
                output_dir=output_dir,
                dataset_type=dataset_type,
                use_rgbd=(dataset_type.lower() == "tum" and bin_name == "tum_rgbd"),
                staged_association_file=staged_association_file,
                staged_timestamps_file=staged_timestamps_file,
            )

        return self._build_container_execution_spec(
            ctx=ctx,
            bin_name=bin_name,
            orb_config=orb_config,
            gaussian_config=gaussian_config,
            dataset_path=dataset_path,
            output_dir=output_dir,
            dataset_type=dataset_type,
            use_rgbd=(dataset_type.lower() == "tum" and bin_name == "tum_rgbd"),
            staged_association_file=staged_association_file,
            staged_timestamps_file=staged_timestamps_file,
        )

    def _build_host_execution_spec(
        self,
        bin_name: str,
        orb_config: Path,
        gaussian_config: Path,
        dataset_path: Path,
        output_dir: Path,
        dataset_type: str,
        use_rgbd: bool,
        staged_association_file: Optional[Path],
        staged_timestamps_file: Optional[Path],
    ) -> Optional[ExecutionSpec]:
        """Build the host (conda/native) execution spec for Photo-SLAM."""
        bin_dir = self.photoslam_path / "bin"
        executable = bin_dir / bin_name

        if dataset_type.lower() == "tum":
            cmd_args = [
                str(executable),
                str(self.photoslam_path / "ORB-SLAM3" / "Vocabulary" / "ORBvoc.txt"),
                str(orb_config),
                str(gaussian_config),
                str(dataset_path.resolve()),
            ]
            if use_rgbd:
                cmd_args.append(str(staged_association_file))
            cmd_args.append(str(output_dir.resolve()) + "/")
            cmd_args.append("no_viewer")
        else:
            cmd_args = [
                str(executable),
                str(self.photoslam_path / "ORB-SLAM3" / "Vocabulary" / "ORBvoc.txt"),
                str(orb_config),
                str(gaussian_config),
                str(dataset_path.resolve()),
                str(staged_timestamps_file),
                str(output_dir.resolve()) + "/",
                "no_viewer",
            ]

        if not executable.exists():
            logger.error(f"Executable not found: {executable}")
            return None

        logger.info(f"  Executable: {executable.name}")
        logger.info(f"  ORB config: {orb_config}")
        logger.info(f"  Gaussian config: {gaussian_config}")

        return ExecutionSpec(
            cmd=cmd_args,
            custom_runner=lambda spec: self._run_photoslam(spec.cmd),
            log_prefix="Photo-SLAM",
        )

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        bin_name: str,
        orb_config: Path,
        gaussian_config: Path,
        dataset_path: Path,
        output_dir: Path,
        dataset_type: str,
        use_rgbd: bool,
        staged_association_file: Optional[Path],
        staged_timestamps_file: Optional[Path],
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``photoslam:latest``.

        The container ships the C++ binaries + the ORB-SLAM3 vocabulary +
        the cfg/ tree at ``_CONTAINER_WORKDIR``.

        ``dataset_path`` (the host's staged dataset root from
        ``_stage_dataset``) is bind-mounted at ``_CONTAINER_DATASET_PATH``.
        Because that staged dir is built from absolute-path symlinks into
        the real dataset / perturbed camera dirs, we ALSO bind-mount each
        symlink target at its same host path inside the container so the
        symlinks resolve correctly. Otherwise Photo-SLAM's C++ binary
        would open broken symlinks and fail to read frames.

        The host-staged association / timestamps file is bind-mounted to
        a stable ``/dataset_meta/`` path so the container can refer to it
        by a path that doesn't collide with the dataset mount.
        """
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            "-v", f"{dataset_path.resolve()}:{_CONTAINER_DATASET_PATH}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
        ]

        # Walk the staged dataset for absolute-path symlinks and bind-mount
        # each symlink target at its same host path inside the container so
        # the symlinks (e.g. staged rgb/ -> real rgb/ in the perturbed dir)
        # resolve. ro-bind so the container can't accidentally mutate the
        # source data.
        for host_path in self._collect_symlink_target_mounts(dataset_path):
            container_cmd.extend(["-v", f"{host_path}:{host_path}:ro"])

        # Bind-mount any host-staged metadata files (TUM associations or
        # EuRoC per-frame timestamps) into a stable container path so the
        # C++ binary can refer to them by a path it knows ahead of time.
        if use_rgbd and isinstance(staged_association_file, Path):
            container_cmd.extend([
                "-v",
                f"{staged_association_file.resolve()}:{_CONTAINER_ASSOC_PATH}:ro",
            ])
        if dataset_type.lower() == "euroc" and isinstance(staged_timestamps_file, Path):
            container_cmd.extend([
                "-v",
                f"{staged_timestamps_file.resolve()}:{_CONTAINER_EUROC_TIMESTAMPS_PATH}:ro",
            ])

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # Photo-SLAM always needs CUDA (LibTorch + cuda_rasterizer); ensure
        # the GPU is always attached. HAMi adds extra env/mounts on top.
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

        # When HAMi is active, pre-create + bind-mount the per-container
        # shared cache file so GpuHamiController.apply() can mutate the
        # cap mid-run by writing directly to the mmap'd region. See
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

        container_cmd.append(self.docker_image)

        # Build the C++ binary invocation inside the container. The image
        # was COPYed into _CONTAINER_WORKDIR and the cfg/ + ORB-SLAM3/
        # tree is shipped relative to that working dir, so we use
        # container-internal paths for those args.
        vocab_path = f"{_CONTAINER_WORKDIR}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
        orb_rel = self._relative_config_in_container(orb_config)
        gauss_rel = self._relative_config_in_container(gaussian_config)
        positional_args = [
            f"{_CONTAINER_WORKDIR}/bin/{bin_name}",
            vocab_path,
            orb_rel,
            gauss_rel,
            _CONTAINER_DATASET_PATH,
        ]
        if use_rgbd:
            positional_args.append(_CONTAINER_ASSOC_PATH)
        if dataset_type.lower() == "euroc":
            positional_args.append(_CONTAINER_EUROC_TIMESTAMPS_PATH)
        positional_args.append(f"{_CONTAINER_OUTPUT_PATH}/")
        positional_args.append("no_viewer")

        # bash -c keeps cd-then-exec semantics consistent with the
        # other Podman wrappers (DROID, VGGT) and lets shell-level
        # env-var expansion / LD_LIBRARY_PATH hooks land in one place
        # if we add them later.
        main_cmd = f"cd {_CONTAINER_WORKDIR} && " + " ".join(positional_args)
        container_cmd.extend(["bash", "-c", main_cmd])

        logger.info(
            "  Executing Photo-SLAM via %s container: %s",
            self.container_runtime, container_name,
        )

        io_target_paths = [str(dataset_path.resolve()), str(output_dir.resolve())]

        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=False,
            log_prefix="Photo-SLAM",
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
        """Build a unique podman-compliant container name for one Photo-SLAM run."""
        raw_name = f"photoslam-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

    def _collect_symlink_target_mounts(self, dataset_root: Path) -> List[str]:
        """Return a deduplicated list of absolute host paths to bind-mount so
        absolute-path symlinks under ``dataset_root`` resolve inside the
        container.

        ``_stage_dataset`` builds the staged dataset by writing absolute
        symlinks pointing into the original perturbed dataset (e.g.
        ``staged_root/rgb -> /abs/path/to/perturbed_rgb``). Inside the
        container, those symlinks point to host-absolute paths that
        aren't visible unless we explicitly bind-mount their target
        directories. This helper walks the staged tree once and returns
        the set of target paths the container needs to see.

        Uses ``os.walk`` with ``followlinks=False`` so we don't recurse
        through symlinks back into their (possibly huge) target trees.

        Each path is returned as a string. Caller bind-mounts each at
        the same path inside the container.
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

    def _relative_config_in_container(self, host_config_path: Path) -> str:
        """Translate a host-side cfg path under ``photoslam_path`` to its
        container-side path under ``_CONTAINER_WORKDIR``.

        The cfg/ tree is COPYed into the image verbatim, so the relative
        layout matches. External configs (anywhere on the host) fall back
        to using their absolute host path inside the container, which
        requires the operator to bind-mount them separately; we log a
        warning so a misconfigured external config doesn't fail silently.
        """
        try:
            relative = host_config_path.resolve().relative_to(self.photoslam_path.resolve())
        except ValueError:
            logger.warning(
                "  External Photo-SLAM config %s is outside of %s; the "
                "container path will not resolve. Bind-mount the config "
                "into %s or move it into Photo-SLAM/cfg/.",
                host_config_path, self.photoslam_path, _CONTAINER_WORKDIR,
            )
            return str(host_config_path.resolve())
        return f"{_CONTAINER_WORKDIR}/{relative.as_posix()}"

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            return False
        try:
            return self._run_execution_spec(spec) == 0
        except Exception as e:
            logger.error(f"Failed to run Photo-SLAM: {e}")
            return False

    def _require_camera_path(
        self,
        request: SLAMRunRequest,
        camera_role: str,
    ) -> Path:
        """Resolve a required camera path from runtime request extras."""
        camera_paths = request.extras.get("camera_paths")
        if not isinstance(camera_paths, dict):
            raise RuntimeError(
                "Photo-SLAM staging requires request.extras['camera_paths'] as a role->path mapping."
            )

        camera_path_value = camera_paths.get(camera_role)
        if not camera_path_value:
            raise RuntimeError(
                f"Photo-SLAM staging requires request.extras['camera_paths']['{camera_role}']."
            )

        camera_path = Path(str(camera_path_value)).resolve()
        if not camera_path.exists() or not camera_path.is_dir():
            raise RuntimeError(
                f"Photo-SLAM camera path for role '{camera_role}' does not exist or is not a directory: {camera_path}"
            )

        return camera_path

    def _get_orb_config(
        self,
        slam_config: str,
        mode: str,
        dataset: str,
        is_external: bool
    ) -> Optional[Path]:
        """Get ORB-SLAM3 config file path."""
        if is_external:
            config_path = Path(slam_config).resolve()
            if config_path.exists():
                return config_path
            logger.error(f"External config not found: {slam_config}")
            return None

        config_dir = self.photoslam_path / "cfg" / "ORB_SLAM3" / mode / dataset
        yaml_candidate = config_dir / f"{slam_config}.yaml"
        if yaml_candidate.exists():
            return yaml_candidate

        exact_candidate = config_dir / slam_config
        if exact_candidate.exists():
            return exact_candidate

        logger.error(
            "ORB config not found for mode=%s dataset=%s requested=%s. Checked: %s, %s",
            mode,
            dataset,
            slam_config,
            yaml_candidate,
            exact_candidate,
        )
        return None

    def _get_gaussian_config(self, slam_config: str, mode: str, dataset: str) -> Optional[Path]:
        """Get Gaussian mapper config file path."""
        config_dir = self.photoslam_path / "cfg" / "gaussian_mapper" / mode / dataset
        yaml_candidate = config_dir / f"{slam_config}.yaml"
        if yaml_candidate.exists():
            return yaml_candidate

        exact_candidate = config_dir / slam_config
        if exact_candidate.exists():
            return exact_candidate

        logger.error(
            "Gaussian config not found for mode=%s dataset=%s requested=%s. Checked: %s, %s",
            mode,
            dataset,
            slam_config,
            yaml_candidate,
            exact_candidate,
        )
        return None

    def _find_existing_tum_association_file(self, dataset_path: Path) -> Optional[Path]:
        """Return an existing TUM association file from dataset root."""
        for name in ("associations.txt", "association.txt", "associate.txt", "assoc.txt"):
            assoc_path = dataset_path / name
            if assoc_path.exists() and assoc_path.is_file():
                return assoc_path
        return None

    def _require_euroc_timestamps_by_frame(self, request: SLAMRunRequest) -> Dict[int, int]:
        """Validate and return EuRoC timestamps from request extras."""
        raw_timestamps = request.extras.get("timestamps_by_frame")
        if not isinstance(raw_timestamps, dict) or not raw_timestamps:
            raise RuntimeError(
                "Photo-SLAM EuRoC staging requires request.extras['timestamps_by_frame']."
            )

        normalized: Dict[int, int] = {}
        for frame_idx, timestamp in raw_timestamps.items():
            if not isinstance(frame_idx, int):
                raise RuntimeError(
                    f"Photo-SLAM invalid timestamps_by_frame key {frame_idx!r}; expected int frame index."
                )
            if frame_idx < 0:
                raise RuntimeError(
                    f"Photo-SLAM invalid timestamps_by_frame key {frame_idx}; frame index must be >= 0."
                )
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise RuntimeError(
                    f"Photo-SLAM invalid timestamp value for frame {frame_idx}: {timestamp!r}."
                )

            if isinstance(timestamp, float):
                if not timestamp.is_integer():
                    raise RuntimeError(
                        "Photo-SLAM EuRoC requires integer-like nanosecond timestamps; "
                        f"frame {frame_idx} has {timestamp!r}."
                    )
                ts_value = int(timestamp)
            else:
                ts_value = int(timestamp)

            normalized[frame_idx] = ts_value

        expected_indices = set(range(len(normalized)))
        actual_indices = set(normalized.keys())
        if actual_indices != expected_indices:
            missing = sorted(expected_indices - actual_indices)
            extra = sorted(actual_indices - expected_indices)
            raise RuntimeError(
                "Photo-SLAM timestamps_by_frame must cover contiguous frame indices 0..N-1 "
                f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"extra={extra[:5]}{'...' if len(extra) > 5 else ''})."
            )

        previous = None
        for frame_idx in range(len(normalized)):
            current = normalized[frame_idx]
            if previous is not None and current <= previous:
                raise RuntimeError(
                    "Photo-SLAM timestamps_by_frame must be strictly increasing by frame index: "
                    f"frame {frame_idx - 1}={previous}, frame {frame_idx}={current}."
                )
            previous = current

        return normalized

    def _list_camera_images(self, image_dir: Path) -> List[Path]:
        """List camera images in deterministic order."""
        image_files = sorted(image_dir.glob("*.png"))
        if not image_files:
            image_files = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.jpeg"))
        if not image_files:
            raise RuntimeError(f"Photo-SLAM found no PNG/JPG images in {image_dir}.")
        return image_files

    def _write_euroc_timestamps_file(
        self,
        timestamps_by_frame: Dict[int, int],
        output_path: Path,
    ) -> None:
        """Write EuRoC timestamps file from validated frame-index mapping."""
        with open(output_path, "w", encoding="utf-8") as file_handle:
            for frame_idx in range(len(timestamps_by_frame)):
                file_handle.write(f"{timestamps_by_frame[frame_idx]}\n")

    def _run_photoslam(self, cmd_args: List[str]) -> bool:
        """Execute Photo-SLAM."""
        logger.info("  Executing Photo-SLAM...")

        try:
            import select
            import time

            env = os.environ.copy()
            existing_paths = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p]
            required_lib_paths = [
                self.photoslam_path / "lib",
                self.photoslam_path / "ORB-SLAM3" / "lib",
                self.photoslam_path / "ORB-SLAM3" / "Thirdparty" / "DBoW2" / "lib",
                self.photoslam_path / "ORB-SLAM3" / "Thirdparty" / "g2o" / "lib",
                self.photoslam_path.parent.parent / "slam-frameworks" / "pyslam" / "thirdparty" / "opencv" / "build_cuda" / "lib",
                self.photoslam_path / "libtorch" / "lib",
                Path.home() / "miniconda3" / "envs" / "S3PO-GS" / "lib",
            ]

            resolved_paths: List[str] = []
            seen = set()
            for path in required_lib_paths:
                path_str = str(path)
                if path.exists() and path_str not in seen:
                    resolved_paths.append(path_str)
                    seen.add(path_str)

            for path in existing_paths:
                if path not in seen:
                    resolved_paths.append(path)
                    seen.add(path)

            env["LD_LIBRARY_PATH"] = ":".join(resolved_paths)

            process = self._spawn_streaming_process(
                cmd_args,
                cwd=self.photoslam_path,
                env=env,
                start_new_session=True,
            )

            shutdown_seen = False
            last_output_time = time.time()
            idle_timeout = 120  # Kill if no output for 2 minutes after shutdown

            while True:
                ret = process.poll()
                if ret is not None:
                    for line in process.stdout:
                        line = line.rstrip()
                        if line:
                            logger.info(f"    [Photo-SLAM] {line}")
                    break

                ready, _, _ = select.select([process.stdout], [], [], 1.0)
                if ready:
                    line = process.stdout.readline()
                    if line:
                        line = line.rstrip()
                        logger.info(f"    [Photo-SLAM] {line}")
                        last_output_time = time.time()
                        if "Shutdown" in line:
                            shutdown_seen = True

                if shutdown_seen and (time.time() - last_output_time) > idle_timeout:
                    logger.error(
                        "  Photo-SLAM reached shutdown but remained idle without exiting; "
                        "killing process and marking run as failed."
                    )
                    self._kill_process_group(process)
                    return False

                if (time.time() - last_output_time) > 7200:
                    logger.error("  Photo-SLAM timed out (no output for 2 hours)")
                    self._kill_process_group(process)
                    return False

            if process.returncode != 0:
                logger.error(f"Photo-SLAM failed with return code {process.returncode}")
                return False

            logger.info("  Photo-SLAM completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to run Photo-SLAM: {e}")
            if 'process' in locals():
                self._kill_process_group(process)
            return False

    def _find_trajectory(self, output_dir: Path) -> Optional[Path]:
        """Find the trajectory file in the output directory."""
        # Photo-SLAM saves trajectories in TUM format
        trajectory_names = [
            "CameraTrajectory_TUM.txt",
            "KeyFrameTrajectory_TUM.txt",
            "CameraTrajectory.txt",
        ]

        for name in trajectory_names:
            traj_path = output_dir / name
            if traj_path.exists():
                standard_path = output_dir / "CameraTrajectory.txt"
                if traj_path != standard_path:
                    shutil.copy(traj_path, standard_path)
                return standard_path

        logger.error(f"No trajectory file found in {output_dir}")
        return None

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw Photo-SLAM trajectory output."""
        return self._find_trajectory(request.output_dir)

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Finalize Photo-SLAM trajectory in standard TUM path."""
        traj_path = raw_trajectory
        # For EuRoC, Photo-SLAM outputs timestamps in seconds, but
        # the evaluation pipeline expects nanoseconds (to match ORB-SLAM3 format)
        if request.dataset_type.lower() == "euroc":
            self._convert_timestamps_to_nanoseconds(traj_path)

        logger.info(f"  Trajectory saved to: {traj_path}")
        return traj_path

    def _convert_timestamps_to_nanoseconds(self, traj_path: Path) -> None:
        """Convert trajectory timestamps from seconds to nanoseconds.

        Photo-SLAM outputs timestamps in seconds for EuRoC datasets, but
        the evaluation pipeline expects nanoseconds (to match ORB-SLAM3 format).
        """
        lines = []
        with open(traj_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    lines.append(line)
                    continue

                parts = line.split()
                if len(parts) >= 8:
                    try:
                        timestamp = float(parts[0])
                        if timestamp < 1e15:
                            timestamp_ns = int(timestamp * 1e9)
                            parts[0] = str(timestamp_ns)
                        lines.append(' '.join(parts))
                    except ValueError:
                        lines.append(line)
                else:
                    lines.append(line)

        with open(traj_path, 'w') as f:
            f.write('\n'.join(lines) + '\n')

        logger.info("  Converted trajectory timestamps to nanoseconds (EuRoC format)")

    def cleanup(self) -> None:
        """Clean up temporary files."""
        if self._staged_dataset_dir and self._staged_dataset_dir.exists():
            try:
                shutil.rmtree(self._staged_dataset_dir)
            except Exception as e:
                logger.warning(f"  Failed to cleanup temp dir: {e}")
        self._staged_dataset_dir = None
