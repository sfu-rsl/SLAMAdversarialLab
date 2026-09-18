"""S3PO-GS (Self-Supervised Photo-realistic Online Gaussian-Splat SLAM).

S3PO-GS is a monocular Gaussian-splat SLAM that uses MASt3R (foundation
pairwise stereo / 3D regression) as a depth + correspondence prior in a
frontend / backend tracker pair on top of a 3D Gaussian-splat map. The
backend / frontend / GUI run in separate ``torch.multiprocessing``
processes. Two execution runtimes are wired into the SAL framework,
mirroring VGGT-SLAM, DROID-SLAM, Photo-SLAM, MASt3R-SLAM, and GigaSLAM:

* ``container_runtime=None`` (default): host conda environment
  ``S3PO-GS`` at ``deps/slam-algorithms/S3PO-GS`` invoked via
  ``python slam.py --config <config.yaml>``.
* ``container_runtime="podman"``: ``s3pogs:latest`` Podman image. This
  path is the one runtime-stress scenarios use to exercise HAMi GPU
  caps (including per-phase runtime mutation) against S3PO-GS's
  MASt3R + Gaussian-splat + simple-knn + diff-gaussian-rasterization
  CUDA hot path. Fourth foundation-model SLAM ported to the
  dual-runtime contract (sibling to VGGT-SLAM, MASt3R-SLAM, and
  GigaSLAM).
"""

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from .config_utils import tee_console_output
from .base import ExecutionSpec, SLAMAlgorithm
from .types import SLAMRunRequest, SensorMode, SLAMRuntimeContext

logger = logging.getLogger(__name__)


_PODMAN_IMAGE = "s3pogs:latest"
# S3PO-GS derives its save-subdir from the LAST three components of
# ``Dataset.dataset_path`` (slam.py:274-277):
#   ``path = config["Dataset"]["dataset_path"].split("/")``
#   ``save_dir = os.path.join(<results>, path[-3]+"_"+path[-2], datetime)``
# We mount the KITTI sequence ROOT (not just rgb/) at
# ``/dataset/KITTI/<NN>`` so that ``dataset_path = "/dataset/KITTI/<NN>/"``
# yields ``path[-3]+"_"+path[-2] = "KITTI_<NN>"``, mirroring the conda
# path's save-subdir.
_CONTAINER_DATASET_PARENT = "/dataset"
_CONTAINER_OUTPUT_PATH = "/output"
# MASt3R weights (~2.5 GB) are downloaded by huggingface_hub from
# ``naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`` on first
# call to ``AsymmetricMASt3R.from_pretrained`` (slam.py:294-295). They
# cache at ``~/.cache/huggingface/hub``. Bind-mount the host's HF cache
# so the weights persist across container runs and don't re-download
# into the ephemeral container layer on every invocation.
_CONTAINER_HF_CACHE_PATH = "/root/.cache/huggingface"
# Torch hub cache mounted for parity with VGGT/DROID/Photo/MASt3R/Giga;
# S3PO-GS itself doesn't currently call torch.hub.load but upstream
# may add it.
_CONTAINER_TORCH_HUB_PATH = "/root/.cache/torch/hub"
_CONTAINER_WORKDIR = "/s3pogs"


class S3POGSAlgorithm(SLAMAlgorithm):
    """S3PO-GS with dual conda+podman runtimes (mono KITTI only)."""

    def __init__(self, container_runtime: Optional[str] = None):
        if container_runtime is not None and container_runtime != "podman":
            raise ValueError(
                f"container_runtime must be None or 'podman', got {container_runtime!r}"
            )
        self.container_runtime = container_runtime
        self.s3pogs_path = Path(__file__).parent.parent.parent / "deps" / "slam-algorithms" / "S3PO-GS"
        self.conda_env = "S3PO-GS"
        self.docker_image = _PODMAN_IMAGE
        self._temp_dataset_link: Optional[Path] = None

    @property
    def runtime_stress_target_kind(self) -> str:
        if self.container_runtime is None:
            return "host_process_group"
        return f"{self.container_runtime}_container"

    @property
    def name(self) -> str:
        return "s3pogs"

    @property
    def supported_datasets(self) -> Dict[str, List[str]]:
        return {
            "kitti": ["mono"],
        }

    def resolve_config_name(self, sequence: str, dataset_type: str, sensor_mode: Optional[SensorMode] = None) -> Optional[str]:
        if dataset_type.lower() == "kitti":
            return str(sequence).zfill(2)
        return None

    def _resolve_internal_config_path(self, ctx: SLAMRuntimeContext) -> Optional[Path]:
        return self.s3pogs_path / "configs" / "mono" / "KITTI" / f"{ctx.internal_config_name}.yaml"

    def _preflight_checks(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Validate S3PO-GS runtime dependencies."""
        if not self.s3pogs_path.exists():
            raise RuntimeError(f"S3PO-GS not found at {self.s3pogs_path}")

        config_file = self._resolve_config_path(ctx)

        if config_file is None:
            raise RuntimeError("S3PO-GS external config path is not resolved in runtime context")

        if not config_file.exists():
            raise RuntimeError(f"S3PO-GS config file not found: {config_file}")

        if self.container_runtime is None:
            # Conda path: rely on conda env existence; we don't probe binaries
            # here (matches DROID/VGGT/Photo/MASt3R/Giga conda-path preflight)
            # — the conda activate will fail loudly if missing.
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
                    self.s3pogs_path,
                    self.docker_image,
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
        """Prepare dataset structure expected by S3PO-GS.

        The conda path uses the staged sequence dir (which contains
        ``rgb/`` and ``calib.txt``) as ``Dataset.dataset_path``. The
        podman path bind-mounts the same staged dir into the container
        and points ``Dataset.dataset_path`` at the container-side path.
        """
        return self._prepare_dataset(
            request.dataset_path,
            ctx.sequence_name,
            request.extras.get("camera_paths", {}),
        )

    def _cleanup_staged_dataset(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> None:
        """Remove temporary dataset staging directory."""
        self._cleanup_temp_dataset_link()

    def _build_execution_inputs(
        self,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Dict[str, object]]:
        """Build resolved runtime inputs used by S3PO-GS execution."""
        prepared_path = ctx.effective_dataset_path
        config_file = self._resolve_config_path(ctx)
        if not prepared_path or config_file is None:
            return None
        return {
            "prepared_path": prepared_path,
            "output_dir": request.output_dir,
            "is_stereo": request.sensor_mode == SensorMode.STEREO,
            "config_file": config_file,
            "sequence_name": ctx.sequence_name,
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
        sequence_name = inputs["sequence_name"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if self.container_runtime is None:
            return ExecutionSpec(
                cmd=["s3pogs"],
                custom_runner=lambda _spec, config_file=config_file: self._run_s3pogs(config_file),
                log_prefix="S3PO-GS",
            )

        return self._build_container_execution_spec(
            ctx=ctx,
            prepared_path=prepared_path,
            host_config=config_file,
            output_dir=output_dir,
            sequence_name=sequence_name,
        )

    def _build_container_execution_spec(
        self,
        ctx: SLAMRuntimeContext,
        prepared_path: Path,
        host_config: Path,
        output_dir: Path,
        sequence_name: str,
    ) -> ExecutionSpec:
        """Build a Podman execution spec that runs ``s3pogs:latest``.

        The container ships the S3PO-GS source tree + the prebuilt
        ``simple-knn`` and ``diff-gaussian-rasterization`` native CUDA
        extensions + the bundled in-tree dust3r / croco / mast3r
        packages.

        ``prepared_path`` (the host's staged KITTI sequence dir from
        ``_prepare_dataset``) is bind-mounted at
        ``/dataset/KITTI/<NN>`` so that S3PO-GS's save-subdir derivation
        (slam.py:274-277, ``path[-3]+"_"+path[-2]``) yields the same
        ``KITTI_<NN>`` label as on the conda path.

        Because that staged dir uses an absolute-path symlink for
        ``rgb/`` pointing into the real perturbed camera dir, we also
        bind-mount that target at its same host path inside the
        container so the symlink resolves.

        The trajectory file is written at
        ``/output/KITTI_<NN>/<datetime>/plot/trj_final.json`` because
        we override ``Results.save_dir`` to ``/output`` in the per-run
        config; the host output_dir is bind-mounted at ``/output`` so
        the trajectory lands directly on the host with no post-exit
        copy.
        """
        container_name = self._build_runtime_stress_container_name(ctx, output_dir)
        torch_hub_cache = Path.home() / ".cache" / "torch" / "hub"
        torch_hub_cache.mkdir(parents=True, exist_ok=True)
        hf_cache = Path.home() / ".cache" / "huggingface"
        hf_cache.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Container-side staged dataset path. Keep the leaf == sequence
        # number so split("/") components match (path[-3]+"_"+path[-2]
        # = "KITTI_<NN>").
        container_dataset_path = (
            f"{_CONTAINER_DATASET_PARENT}/KITTI/{sequence_name.zfill(2)}"
        )

        # Generate the per-run config inheriting from the chosen
        # configs/mono/KITTI/<NN>.yaml; override Dataset.dataset_path
        # and Results.save_dir so the container reads container-side
        # paths.
        config_file = self._create_config(
            base_config=host_config,
            dataset_path_value=container_dataset_path + "/",
            save_dir_value=_CONTAINER_OUTPUT_PATH,
            output_dir=output_dir,
        )
        if config_file is None:
            raise RuntimeError("S3PO-GS podman path: failed to create per-run config")
        host_config_file = config_file.resolve()
        # output_dir is bind-mounted at /output; the per-run config
        # is written to output_dir/s3pogs_config.yaml, so its container path
        # is deterministic.
        container_config = f"{_CONTAINER_OUTPUT_PATH}/{host_config_file.name}"

        container_cmd: List[str] = [
            self.container_runtime, "run", "--rm",
            "--name", container_name,
            # S3PO-GS uses torch.multiprocessing.set_start_method("spawn")
            # (slam.py:233) and spawns frontend/backend/gui worker
            # processes that exchange torch IPC tensors over mp.Queue
            # (slam.py:70-74, 110-113). These tensors back to /dev/shm.
            # Podman's default 64 MB /dev/shm can SIGBUS the workers
            # right after spawn with no Python traceback. 8 GB matches
            # the per-keyframe Gaussian/depth tensor working set plus
            # IPC headroom (parity with DROID/Photo/MASt3R/Giga).
            "--shm-size=8g",
            "-v", f"{prepared_path.resolve()}:{container_dataset_path}:ro",
            "-v", f"{output_dir.resolve()}:{_CONTAINER_OUTPUT_PATH}",
            # MASt3R foundation-model weights cache (~2.5 GB) bind-mounted
            # so first run doesn't have to re-download on every container
            # invocation.
            "-v", f"{hf_cache.resolve()}:{_CONTAINER_HF_CACHE_PATH}",
            "-v", f"{torch_hub_cache.resolve()}:{_CONTAINER_TORCH_HUB_PATH}",
        ]

        # The staged dataset dir's rgb/ entry is a symlink to the
        # perturbed camera dir (absolute host path); bind-mount that
        # target so the symlink resolves inside the container.
        for host_path in self._collect_symlink_target_mounts(prepared_path):
            container_cmd.extend(["-v", f"{host_path}:{host_path}:ro"])

        extras = self._runtime_stress_launch_extras()
        extras_devices = list(extras.get("devices") or [])
        # S3PO-GS always needs CUDA (MASt3R foundation model +
        # Gaussian-splat rasterizer + simple-knn); ensure the GPU is
        # always attached. HAMi adds extra env/mounts on top.
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

        # SAL real-time deadline harness: propagate SAL_DEADLINE_* + the
        # iterator dir (no-op when off), and override the frontend sub-module
        # (the frame loop is in utils/slam_frontend.py, not slam.py) with the
        # host copy so the deadline hooks reach the SLAM without a rebuild.
        from ..runtime_stress.podman_injection import (
            apply_entrypoint_override,
            apply_realtime_to_podman_cmd,
        )
        apply_realtime_to_podman_cmd(container_cmd, _CONTAINER_OUTPUT_PATH)
        apply_entrypoint_override(
            container_cmd,
            host_path=Path(__file__).resolve().parents[2]
            / "deps" / "slam-algorithms" / "S3PO-GS" / "utils" / "slam_frontend.py",
            container_path=f"{_CONTAINER_WORKDIR}/utils/slam_frontend.py",
        )

        container_cmd.append(self.docker_image)

        # Run S3PO-GS's slam.py entrypoint. Force PYTHONUNBUFFERED so
        # the streaming output reaches the SAL logs in real time.
        # ``inherit_from`` in the generated config is a relative path
        # ("configs/mono/KITTI/base_config.yaml") which resolves
        # against CWD; cd into /s3pogs first so the inherit lookup
        # finds the base config bundled in the image.
        main_cmd = (
            f"cd {_CONTAINER_WORKDIR} && "
            f"PYTHONUNBUFFERED=1 python -u slam.py "
            f"--config {container_config}"
        )
        container_cmd.extend(
            ["bash", "-c", tee_console_output(main_cmd, _CONTAINER_OUTPUT_PATH)]
        )

        logger.info(
            "  Executing S3PO-GS via %s container: %s",
            self.container_runtime, container_name,
        )

        io_target_paths = [str(prepared_path.resolve()), str(output_dir.resolve())]

        # Use a custom_runner so we can mirror the conda path's
        # ``stop_on_line`` + force-kill semantics for the podman path:
        # S3PO-GS's slam.py finishes the SLAM and prints "Total FPS"
        # but then hangs in ``backend_process.join()`` because the
        # backend mp.Process doesn't exit cleanly (well-known upstream
        # bug; the conda wrapper works around it the same way). Without
        # this, the container stays alive after "Total FPS" until the
        # default timeout kills it (~2 hours).
        return ExecutionSpec(
            cmd=container_cmd,
            stream_output=True,
            log_prefix="S3PO-GS",
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
            custom_runner=lambda _spec, cmd=container_cmd, container_name=container_name, io_target_paths=io_target_paths, log_prefix="S3PO-GS": self._run_container_with_total_fps_stop(
                cmd=cmd,
                container_name=container_name,
                io_target_paths=io_target_paths,
                log_prefix=log_prefix,
            ),
        )

    def _run_container_with_total_fps_stop(
        self,
        cmd: List[str],
        container_name: str,
        io_target_paths: List[str],
        log_prefix: str,
    ) -> bool:
        """Run the S3PO-GS container, stream its output, and stop+kill the
        container as soon as ``"Total FPS"`` appears.

        Mirrors the conda runner's ``stop_on_line`` workaround. Without
        this, slam.py hangs in ``backend_process.join()`` after the
        evaluation completes (mp.Process worker doesn't exit cleanly;
        well-known upstream issue) and the container stays alive until
        the wrapper's default timeout (~2 hours) kicks in.
        """
        process = self._spawn_streaming_process(
            cmd,
            target_kind=f"{self.container_runtime}_container",
            target_metadata={
                "container_name": container_name,
                "io_target_paths": io_target_paths,
            },
        )

        completed = False
        crashed = False

        # Exceptions that are TERMINAL for S3PO-GS, i.e. raised inside eval_ate,
        # after which the process can never reach its "Total FPS" line and hangs
        # in mp.Process cleanup instead.
        #
        # This list is deliberately SPECIFIC rather than "any traceback". Not
        # every traceback is fatal -- one printed from an except block is
        # followed by a normal finish -- and stopping on those would cut short a
        # run that was going to succeed. Only add an entry here after observing
        # that it actually leaves the process hung.
        #
        # GeometryException: evo's umeyama alignment on a degenerate trajectory.
        # Observed twice with warmup_frames=0, where s3pogs received 2-5
        # frames out of 200.
        TERMINAL_ERRORS = ("GeometryException",)

        def _stop_on_line(line: str) -> bool:
            nonlocal completed, crashed
            if "Total FPS" in line:
                completed = True
                logger.info(
                    "  S3PO-GS evaluation complete (Total FPS line seen); "
                    "stopping container '%s' so the mp.Process cleanup "
                    "hang doesn't block the next run...",
                    container_name,
                )
                return True
            # MUST stop from inside this callback, not after the stream ends.
            # The container stays alive holding the pipe open, so the streaming
            # loop never returns and any handling placed after it is
            # unreachable. A first attempt at this fix put the recovery after
            # _stream_process_output and it never executed.
            if any(err in line for err in TERMINAL_ERRORS):
                crashed = True
                logger.error(
                    "  S3PO-GS hit a TERMINAL error before eval_ate finished: "
                    "%s. It cannot reach its 'Total FPS' line from here and will "
                    "hang in mp.Process cleanup, so container '%s' is being "
                    "stopped now. THIS RUN IS A FAILURE and is reported as one.",
                    line.strip()[:160], container_name,
                )
                return True
            return False

        stopped_early = self._stream_process_output(
            process, log_prefix, stop_on_line=_stop_on_line,
        )

        if stopped_early or completed or crashed:
            # Kill the container; on the completion path the bind-mounted
            # /output already has the final trajectory written by eval_ate,
            # which runs immediately before the "Total FPS" Log() call.
            self._stop_container(container_name, runtime=self.container_runtime)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                process.wait()
            # A terminal error is a FAILED run, never a completed one. Stopping
            # the hang must not launder the crash into a success: the traceback
            # stays in the log and the health gate scores this failed.
            return not crashed

        # No "Total FPS" -> SLAM crashed before reaching eval_ate.
        #
        # Measured: a crashed S3PO-GS does not exit on its own. It hangs in the
        # same mp.Process cleanup the completion path works around, so the
        # 7200 s wait below runs to the full two hours. With warmup_frames=0
        # this fired for real: s3pogs received 2-5 frames, evo's
        # umeyama alignment raised GeometryException on the degenerate
        # trajectory, and the container then sat until the campaign's own
        # timeout killed it, costing 3 of its 6 rungs.
        #
        # So when a traceback was seen, give the container a short grace to exit
        # on its own and then stop it. This does NOT paper over the crash: the
        # traceback stays in the log, this returns False, and the run is scored
        # a failure by the health gate as it should be. It only declines to wait
        # two hours for a process already known to be hung.
        wait_s = 120 if crashed else 7200
        if crashed:
            logger.error(
                "  S3PO-GS crashed before eval_ate (traceback seen and no "
                "'Total FPS' line). Waiting %ds for a clean exit, then stopping "
                "container '%s' rather than blocking on the mp.Process hang. "
                "This run is a FAILURE and is reported as one.",
                wait_s, container_name,
            )
        self._wait_for_process(process, timeout_seconds=wait_s)
        if process.poll() is None:
            self._stop_container(container_name, runtime=self.container_runtime)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    process.kill()
                except Exception:
                    pass
                process.wait()
        if process.returncode != 0:
            logger.error("S3PO-GS container exited with rc=%s", process.returncode)
            return False
        return True

    def _build_runtime_stress_container_name(
        self,
        ctx: SLAMRuntimeContext,
        output_dir: Path,
    ) -> str:
        """Build a unique podman-compliant container name for one S3PO-GS run."""
        raw_name = f"s3pogs-{ctx.sequence_name}-{output_dir.name}-{uuid.uuid4().hex[:8]}"
        sanitized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", raw_name.lower()).strip("-")
        return sanitized[:120]

    def _collect_symlink_target_mounts(self, dataset_root: Path) -> List[str]:
        """Return absolute host paths to bind-mount so absolute-path
        symlinks under ``dataset_root`` resolve inside the container.

        ``_prepare_dataset`` builds the staged KITTI dir by writing an
        absolute symlink ``rgb -> /abs/path/to/perturbed_rgb``. Inside
        the container, that symlink points to a host-absolute path the
        container can't see unless we bind-mount its target. This
        helper walks the staged tree once and returns the set of
        target paths the container needs.
        """
        if not dataset_root.exists():
            return []

        seen: List[str] = []
        seen_set: set = set()

        def _maybe_record(entry: Path) -> None:
            if not entry.is_symlink():
                return
            try:
                target = Path(os.readlink(entry))
            except OSError:
                return
            if not target.is_absolute():
                return
            target_str = str(target.resolve())
            if target_str in seen_set:
                return
            if not Path(target_str).exists():
                return
            seen.append(target_str)
            seen_set.add(target_str)

        for entry in dataset_root.iterdir():
            _maybe_record(entry)
        for root, dirs, files in os.walk(str(dataset_root), followlinks=False):
            root_path = Path(root)
            for d in dirs:
                _maybe_record(root_path / d)
            for f in files:
                _maybe_record(root_path / f)
        return seen

    def _create_config(
        self,
        base_config: Path,
        dataset_path_value: str,
        save_dir_value: str,
        output_dir: Path,
    ) -> Optional[Path]:
        """Generate a per-run S3PO-GS config inheriting from the chosen base.

        The generated config is written to ``output_dir/s3pogs_config.yaml``
        so it's reachable from inside the container (which bind-mounts
        ``output_dir`` at ``/output``).

        Args:
            base_config: Path to the upstream KITTI config to inherit
                from (e.g. ``configs/mono/KITTI/04.yaml``).
            dataset_path_value: String to write into
                ``Dataset.dataset_path``. For podman this is the
                container-side path (e.g. ``/dataset/KITTI/04/``).
            save_dir_value: String to write into ``Results.save_dir``.
                For podman this is ``/output``.
            output_dir: Host output dir (where the generated YAML is
                written so it can be reached via the /output bind-mount).
        """
        if base_config is None or not base_config.exists():
            logger.error(f"Base config not found: {base_config}")
            return None

        try:
            # We DON'T read+rewrite the base; we write a thin override
            # YAML that ``inherit_from``s the base. This keeps the
            # generated file minimal and avoids reproducing every
            # upstream key (S3PO-GS uses load_config which recursively
            # merges inherited configs — utils/config_utils.py).
            # ``inherit_from`` is resolved by slam.py relative to CWD;
            # container CWD is /s3pogs and the base config ships inside
            # the image at the same relative tree, so use the relative
            # path from the s3pogs source root.
            try:
                rel = base_config.resolve().relative_to(self.s3pogs_path.resolve())
                inherit_value = str(rel)
            except ValueError:
                inherit_value = str(base_config.resolve())

            generated = (
                f'inherit_from: "{inherit_value}"\n'
                f'\n'
                f'Dataset:\n'
                f'  dataset_path: "{dataset_path_value}"\n'
                f'\n'
                f'Results:\n'
                f'  save_dir: "{save_dir_value}"\n'
            )

            temp_config = output_dir / "s3pogs_config.yaml"
            with open(temp_config, 'w') as f:
                f.write(generated)

            logger.info(f"  Created config: {temp_config}")
            return temp_config

        except Exception as e:
            raise RuntimeError(f"Failed to create S3PO-GS config: {e}") from e

    def _execute(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> bool:
        inputs = ctx.execution_inputs
        output_dir = inputs["output_dir"]
        is_stereo = inputs["is_stereo"]
        config_file = inputs["config_file"]

        output_dir.mkdir(parents=True, exist_ok=True)

        if is_stereo:
            logger.warning("  S3PO-GS only supports monocular mode. Stereo flag ignored.")

        logger.info(f"  Using config: {config_file}")

        spec = self._build_execution_spec(request, ctx)
        if spec is None:
            ctx.notes["execution_success"] = False
            return False
        try:
            ctx.notes["execution_started_at"] = time.time()
            execution_success = self._run_execution_spec(spec) == 0
            ctx.notes["execution_success"] = execution_success
            return execution_success
        except Exception as e:
            ctx.notes["execution_success"] = False
            logger.error(f"Failed to run S3PO-GS: {e}")
            return False

    def _prepare_dataset(
        self,
        dataset_path: Path,
        sequence: str,
        camera_paths: Dict[str, object],
    ) -> Optional[Path]:
        """Prepare dataset for S3PO-GS format (rgb/, poses.txt, calib.txt).

        For the conda path, the staged sequence dir sits under
        ``self.s3pogs_path/datasets/KITTI/<NN>`` so slam.py's relative
        ``dataset_path`` (default ``datasets/KITTI/04/``) lands there
        with no further config rewriting.

        For the podman path, the same dir is generated and then
        bind-mounted into the container at ``/dataset/KITTI/<NN>``.
        The wrapper still places the staged dir under
        ``self.s3pogs_path/datasets`` so it's distinct from any other
        run's staging and gets cleaned up by ``_cleanup_temp_dataset_link``.
        """
        logger.info("  Preparing dataset for S3PO-GS...")

        sequence_name = sequence.zfill(2)
        s3pogs_datasets = self.s3pogs_path / "datasets" / "KITTI"
        s3pogs_datasets.mkdir(parents=True, exist_ok=True)
        s3pogs_sequence = s3pogs_datasets / sequence_name

        if s3pogs_sequence.exists() or s3pogs_sequence.is_symlink():
            if s3pogs_sequence.is_symlink():
                s3pogs_sequence.unlink()
            else:
                shutil.rmtree(s3pogs_sequence)

        s3pogs_sequence.mkdir(parents=True, exist_ok=True)
        self._temp_dataset_link = s3pogs_sequence

        left_image_path = self._resolve_left_camera_path(camera_paths)
        if left_image_path is None:
            return None

        rgb_link = s3pogs_sequence / "rgb"

        try:
            os.symlink(str(left_image_path.resolve()), str(rgb_link))
        except (OSError, NotImplementedError) as e:
            logger.error(f"Failed to create symlink: {e}")
            return None

        # S3PO-GS expects poses.txt in KITTI format (12 values per line)
        poses_src = dataset_path / "poses.txt"
        poses_dst = s3pogs_sequence / "poses.txt"

        if poses_src.exists():
            shutil.copy2(poses_src, poses_dst)
        else:
            poses_dir = dataset_path / "poses"
            if poses_dir.exists():
                pose_files = list(poses_dir.glob("*.txt"))
                if pose_files:
                    shutil.copy2(pose_files[0].resolve(), poses_dst)
                else:
                    logger.error(f"No pose files found in {poses_dir}")
                    return None
            else:
                # Try multiple locations for KITTI poses
                kitti_poses_candidates = [
                    # Parent structure: ../../poses/{name}.txt
                    dataset_path.parent.parent / "poses" / f"{sequence_name}.txt",
                    # Project base datasets/kitti/poses/{name}.txt
                    Path(__file__).parent.parent.parent / "datasets" / "kitti" / "poses" / f"{sequence_name}.txt",
                ]

                poses_found = False
                for kitti_poses in kitti_poses_candidates:
                    if kitti_poses.exists():
                        shutil.copy2(kitti_poses, poses_dst)
                        logger.info(f"    Copied poses from: {kitti_poses}")
                        poses_found = True
                        break

                if not poses_found:
                    logger.error(f"poses.txt not found in {dataset_path} or any standard KITTI location")
                    return None

        calib_src = dataset_path / "calib.txt"
        if calib_src.exists():
            shutil.copy2(calib_src, s3pogs_sequence / "calib.txt")

        logger.info(f"    Dataset prepared at: {s3pogs_sequence}")
        return s3pogs_sequence

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

    def _run_s3pogs(self, config_file: Path) -> bool:
        """Execute S3PO-GS SLAM on the host conda env."""
        logger.info("  Executing S3PO-GS...")

        conda_init = "source ~/miniconda3/etc/profile.d/conda.sh"
        if not Path(os.path.expanduser("~/miniconda3")).exists():
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"

        cmd = (
            f"cd {self.s3pogs_path} && "
            f"{conda_init} && "
            f"conda activate {self.conda_env} && "
            f"python slam.py --config {config_file}"
        )

        try:
            process = self._spawn_streaming_process(
                ["bash", "-c", cmd],
                start_new_session=True  # Create new process group for clean termination
            )

            completed = False
            def _stop_on_line(line: str) -> bool:
                nonlocal completed
                if "Total FPS" in line:
                    completed = True
                    logger.info("  S3PO-GS evaluation complete, terminating process...")
                    return True
                return False

            self._stream_process_output(process, "S3PO-GS", stop_on_line=_stop_on_line)

            # Force terminate - S3PO-GS multiprocessing doesn't exit cleanly
            if completed:
                import signal
                try:
                    # Kill the process group to ensure all child processes are terminated
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
                # Give it a moment then force kill if needed
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                logger.info("  S3PO-GS completed successfully")
                return True

            self._wait_for_process(process, timeout_seconds=7200)

            if process.returncode != 0:
                logger.error(f"S3PO-GS failed with return code {process.returncode}")
                return False

            logger.info("  S3PO-GS completed successfully")
            return True

        except subprocess.TimeoutExpired:
            process.kill()
            logger.error("S3PO-GS timed out after 2 hours")
            return False
        except Exception as e:
            logger.error(f"Failed to run S3PO-GS: {e}")
            return False

    def _find_raw_trajectory(self, request: SLAMRunRequest, ctx: SLAMRuntimeContext) -> Optional[Path]:
        """Find raw S3PO-GS trajectory output.

        Conda path: S3PO-GS writes the trajectory to
        ``<s3pogs_path>/results/<KITTI_NN>/<datetime>/plot/trj_final.json``
        (Results.save_dir defaults to relative "results"; with CWD =
        s3pogs_path this lands inside the source tree).

        Podman path: the per-run config overrides Results.save_dir to
        ``/output`` (bind-mounted to host ``output_dir``), so the
        trajectory lands at
        ``<output_dir>/<KITTI_NN>/<datetime>/plot/trj_final.json``.
        """
        if not ctx.notes.get("execution_success", False):
            logger.error("S3PO-GS execution did not complete successfully; skipping trajectory discovery")
            return None

        if self.container_runtime is None:
            results_dir = self.s3pogs_path / "results"
        else:
            results_dir = request.output_dir

        if not results_dir.exists():
            logger.error(f"Results directory not found: {results_dir}")
            return None

        run_started_at = float(ctx.notes.get("execution_started_at", 0.0))
        latest_run = None
        latest_time = None

        for dataset_dir in results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            for run_dir in dataset_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                trj_file = run_dir / "plot" / "trj_final.json"
                if trj_file.exists():
                    mtime = trj_file.stat().st_mtime
                    if run_started_at and mtime + 1e-6 < run_started_at:
                        continue
                    if latest_time is None or mtime > latest_time:
                        latest_time = mtime
                        latest_run = trj_file

        if not latest_run:
            logger.error("No trajectory file found in S3PO-GS results")
            return None

        logger.info(f"    Found trajectory: {latest_run}")
        return latest_run

    def _convert_raw_trajectory_to_tum(
        self,
        raw_trajectory: Path,
        request: SLAMRunRequest,
        ctx: SLAMRuntimeContext,
    ) -> Optional[Path]:
        """Convert S3PO-GS trajectory JSON to TUM format with KITTI timestamps."""
        output_dir = request.output_dir
        timestamps_by_frame = self._resolve_timestamps_by_frame(request)

        logger.info("  Converting trajectory to TUM format...")

        try:
            with open(raw_trajectory, 'r') as f:
                data = json.load(f)

            frame_ids = data["trj_id"]
            poses = data["trj_est"]

            # TUM format with actual KITTI timestamps
            # S3PO-GS outputs sparse keyframes - we map frame IDs to real timestamps
            tum_lines = []
            for frame_id, pose_4x4 in zip(frame_ids, poses):
                frame_idx = int(frame_id)
                if frame_idx not in timestamps_by_frame:
                    raise ValueError(
                        f"Missing timestamp for frame index {frame_idx} in timestamps_by_frame."
                    )
                timestamp = timestamps_by_frame[frame_idx]

                pose = np.array(pose_4x4)
                tx, ty, tz = pose[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(pose[:3, :3]).as_quat()
                tum_lines.append(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}")

            tum_path = output_dir / "CameraTrajectory.txt"
            with open(tum_path, 'w') as f:
                f.write('\n'.join(tum_lines))

            logger.info(f"    Converted {len(tum_lines)} keyframe poses to TUM format")
            return tum_path

        except Exception as e:
            logger.error(f"Failed to convert trajectory: {e}")
            return None

    def cleanup(self) -> None:
        """Remove temporary dataset symlink."""
        self._cleanup_temp_dataset_link()

    def _cleanup_temp_dataset_link(self) -> None:
        """Remove temporary S3PO-GS dataset directory/link when present."""
        if self._temp_dataset_link and self._temp_dataset_link.exists():
            try:
                if self._temp_dataset_link.is_symlink():
                    self._temp_dataset_link.unlink()
                else:
                    shutil.rmtree(self._temp_dataset_link)
            except Exception as e:
                logger.warning(f"  Failed to cleanup: {e}")

        self._temp_dataset_link = None
