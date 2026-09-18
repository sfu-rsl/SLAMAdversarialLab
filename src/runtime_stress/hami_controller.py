"""HAMi-core GPU isolation controller.

HAMi enforces a hard VRAM cap and a soft SM throttle by preloading
``libvgpu.so`` and intercepting CUDA driver calls. The cap value
``CUDA_DEVICE_MEMORY_LIMIT`` is read from the env var once at process
init and *seeded* into a writable mmap-backed shared region
(``/tmp/cudevshr.cache``). Subsequent allocations consult the live value
in shared memory on every ``cuMemAlloc``.

This controller exploits that to mutate GPU caps mid-run: bind-mounting
the shared region as a per-container host file lets ``apply()`` write a
new ``uint64`` to ``shared_region_t.limit[0]`` (and ``sm_limit[0]``) at
phase boundaries. HAMi reads it on the next allocation check.

Limitations:
* Tightening the cap rejects future allocations only. HAMi never
  reclaims memory the SLAM has already claimed.
* The byte offsets here are pinned to upstream HAMi-core commit
  ``94fff568c1ccb32cdbd0f2b51de6d12d2902f074`` (the GIT_HASH_94fff56 our
  shipped libvgpu.so is built from). If the binary is rebuilt against a
  newer HAMi version, offsets must be re-derived; the version-mismatch
  check in ``_verify_version`` warns when this happens.

v1 only accepts container target kinds. Host-process support lands
alongside the PhotoSLAM Podman migration.
"""

from __future__ import annotations

import logging
import os
import shutil
import struct
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .controllers import ResourceController
from .models import GpuControl, RuntimeStressControls

logger = logging.getLogger(__name__)

HAMI_LIB_HOST_PATH = os.environ.get("SAL_HAMI_LIB_HOST_PATH", "/opt/hami/libvgpu.so")
HAMI_LIB_CONTAINER_PATH = "/opt/hami/libvgpu.so"
HAMI_LOCK_DIR = "/tmp/vgpulock"
HAMI_SHARED_CACHE = "/tmp/cudevshr.cache"

# Per-container host-side cache files live here. One file per container,
# bind-mounted onto the container's HAMI_SHARED_CACHE so the host can
# pwrite() new caps directly to the mmap that libvgpu.so reads on every
# cuMemAlloc.
HAMI_CACHE_HOST_DIR = Path(
    os.environ.get(
        "SAL_HAMI_CACHE_HOST_DIR",
        str(Path.home() / ".cache" / "sal" / "hami"),
    )
)

# DWARF-extracted byte offsets within shared_region_t for our shipped
# libvgpu.so (HAMi-core commit 94fff56).
SHARED_REGION_SIZE_BYTES = 2009496
INITIALIZED_FLAG_OFFSET = 0
MAJOR_VERSION_OFFSET = 4
MINOR_VERSION_OFFSET = 8
LIMIT_OFFSET = 1632
SM_LIMIT_OFFSET = 1760

EXPECTED_MAJOR_VERSION = 1
EXPECTED_MINOR_VERSION = 1

INIT_WAIT_TIMEOUT_S = 10.0
INIT_POLL_INTERVAL_S = 0.1

_SUPPORTED_TARGET_KINDS = {"docker_container", "podman_container"}


def hami_launch_env(gpu_control: GpuControl) -> Dict[str, str]:
    """Build the HAMi env vars that the SLAM process must inherit.

    Only keys with real values are emitted so callers can merge the
    dict unconditionally without clobbering defaults.
    """
    env: Dict[str, str] = {}

    env["LD_PRELOAD"] = HAMI_LIB_CONTAINER_PATH

    if gpu_control.vram_limit_mb is not None:
        env["CUDA_DEVICE_MEMORY_LIMIT"] = f"{int(gpu_control.vram_limit_mb)}m"

    if gpu_control.sm_limit_percent is not None:
        env["CUDA_DEVICE_SM_LIMIT"] = str(int(gpu_control.sm_limit_percent))

    return env


def hami_launch_mounts() -> List[Tuple[str, str, str]]:
    """Host mounts that must be bound into the container for HAMi to work.

    Reads ``SAL_HAMI_LIB_HOST_PATH`` at call time so tests and dev
    workflows can point at a user-writable copy without sudo.
    """
    host_path = os.environ.get("SAL_HAMI_LIB_HOST_PATH", HAMI_LIB_HOST_PATH)
    return [(host_path, HAMI_LIB_CONTAINER_PATH, "ro")]


def hami_launch_devices() -> List[str]:
    """CDI device refs the container must declare for libvgpu.so to link.

    ``libvgpu.so`` has a ``DT_NEEDED`` dependency on ``libcuda.so.1``, which
    ships only with the NVIDIA driver. Non-CUDA images (ORB-SLAM3) don't
    have it; CDI injection via ``--device nvidia.com/gpu=all`` bind-mounts
    the host driver libs into the container so ``LD_PRELOAD`` resolves.
    """
    return ["nvidia.com/gpu=all"]


def hami_cache_host_path(container_name: str) -> Path:
    """Per-container host path for the bind-mounted HAMi shared cache file.

    Each runtime-stress run gets its own cache file so concurrent runs
    don't fight over the same mmap. The wrapper pre-creates this file
    before launching the container; the controller writes to it at
    phase boundaries.
    """
    if not container_name or not container_name.strip():
        raise ValueError("hami_cache_host_path requires a non-empty container_name")
    return HAMI_CACHE_HOST_DIR / f"cudevshr.{container_name.strip()}"


def prepare_hami_cache_file(container_name: str) -> Path:
    """Pre-create the per-container HAMi shared cache file at the correct size.

    Must be called *before* the container starts, because podman bind
    mounts require the source file to exist. libvgpu.so will populate
    the file with ``shared_region_t`` fields during its init pass.

    Returns the host path. Mode is 0o666 so the rootless-container's
    libvgpu.so can write to it under user-namespace remapping.
    """
    HAMI_CACHE_HOST_DIR.mkdir(parents=True, exist_ok=True)
    path = hami_cache_host_path(container_name)
    if path.exists():
        # Stale from a prior aborted run. Truncate to the right size.
        path.unlink()
    with path.open("wb") as f:
        f.truncate(SHARED_REGION_SIZE_BYTES)
    path.chmod(0o666)
    return path


def verify_cap_enforcement(
    container_name: str,
    expected_mb: int,
    *,
    runtime: str = "podman",
) -> None:
    """Log a warning if the in-container VRAM total does not match the cap.

    Best-effort: if ``nvidia-smi`` isn't present in the container, just
    log at INFO and move on. This is a sanity hook for CUDA workloads;
    for non-CUDA images (like ORB-SLAM3) the check is skipped entirely.
    """
    try:
        result = subprocess.run(
            [
                runtime,
                "exec",
                container_name,
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except Exception as exc:
        logger.info(
            "HAMi cap verification skipped for container '%s' (exec failed: %s)",
            container_name,
            exc,
        )
        return

    if result.returncode != 0:
        logger.info(
            "HAMi cap verification skipped for container '%s': nvidia-smi "
            "not available inside container (returncode=%s).",
            container_name,
            result.returncode,
        )
        return

    raw = (result.stdout or "").strip().splitlines()
    if not raw:
        return

    try:
        reported = int(raw[0].strip())
    except ValueError:
        return

    # HAMi rounds; accept ±5% drift.
    if abs(reported - expected_mb) > max(32, int(expected_mb * 0.05)):
        logger.error(
            "HAMi VRAM cap mismatch for container '%s': expected ~%d MiB, "
            "nvidia-smi reports %d MiB. CUDA version likely mismatched with "
            "the libvgpu.so build — pick a different HAMi release.",
            container_name,
            expected_mb,
            reported,
        )


class GpuHamiController(ResourceController):
    """GPU isolation controller backed by HAMi-core's LD_PRELOAD shim.

    HAMi seeds its caps from env vars at process init, but the live cap
    state is held in a writable mmap-backed shared region. This controller
    mutates that region directly at phase boundaries, so per-phase GPU
    caps are honored even though HAMi never re-reads its env vars.

    See the module docstring for the full mechanism.
    """

    def __init__(self, lib_path: str = HAMI_LIB_HOST_PATH, lock_dir: str = HAMI_LOCK_DIR):
        self._lib_path = Path(lib_path)
        self._lock_dir = Path(lock_dir)
        self._process = None
        self._target_kind: Optional[str] = None
        self._target_metadata: Dict[str, Any] = {}
        self._cache_host_path: Optional[Path] = None
        self._init_observed: bool = False
        self._version_checked: bool = False
        # Last (vram_bytes, sm_percent) confirmed to have landed in the shared
        # region. None means no mutation has succeeded yet, so the launch-time
        # env cap is still what the container is running under.
        self._last_applied: Optional[Tuple[int, int]] = None

    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target_kind == "host_process_group":
            raise RuntimeError(
                "GpuHamiController does not support host_process_group targets in v1. "
                "Host-process GPU isolation is deferred to the PhotoSLAM Podman "
                "migration cycle."
            )
        if target_kind not in _SUPPORTED_TARGET_KINDS:
            raise RuntimeError(
                f"GpuHamiController only supports container targets, got '{target_kind}'."
            )

        if not self._lib_path.exists():
            raise RuntimeError(
                f"HAMi library not found at '{self._lib_path}'. "
                "Follow \"HAMi GPU Isolation\" in README.md to build libvgpu.so "
                "and place it there."
            )

        try:
            self._lock_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(
                f"HAMi lock directory '{self._lock_dir}' could not be created: {exc}"
            ) from exc

        self._remove_legacy_shared_cache()

        self._process = process
        self._target_kind = target_kind
        self._target_metadata = dict(target_metadata or {})

        container_name = self._target_metadata.get("container_name")
        if isinstance(container_name, str) and container_name.strip():
            cache_path = hami_cache_host_path(container_name)
            if cache_path.exists():
                self._cache_host_path = cache_path
            else:
                # Non-CUDA SLAM wrappers (e.g., ORB-SLAM3) legitimately don't
                # pre-create a per-container cache file because they never call
                # cuMemAlloc. apply() will no-op and the cap stays at the
                # launch-time-seeded value — that's the legacy behavior, not a
                # bug. Log at INFO so it's discoverable without being noisy.
                logger.info(
                    "HAMi runtime mutation not configured for container '%s' "
                    "(no per-container cache file at '%s'). The cap will stay "
                    "at the launch-time-seeded value for this run.",
                    container_name,
                    cache_path,
                )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._cache_host_path is None:
            # No host-side cache file means no runtime mutation. The
            # launch-time env-injected cap stays in force. This matches
            # the legacy behavior pre-runtime-mutation.
            #
            # Logged rather than returned silently: for a single-phase run the
            # env cap is the right cap, but for a ladder every rung after the
            # first would quietly inherit phase 0's limit, and until this line
            # existed that left no trace anywhere to detect it by.
            logger.info(
                "HAMi runtime mutation unavailable for container '%s' (no host-side "
                "cache file); the launch-time env cap remains in force for this phase.",
                self._target_metadata.get("container_name"),
            )
            return

        gpu = controls.gpu
        new_vram_bytes = 0
        new_sm_percent = 0
        if gpu is not None:
            if gpu.vram_limit_mb is not None:
                new_vram_bytes = int(gpu.vram_limit_mb) * 1024 * 1024
            if gpu.sm_limit_percent is not None:
                new_sm_percent = int(gpu.sm_limit_percent)

        if not self._init_observed and not self._wait_for_init():
            # Skipping is only safe while the launch-time env cap is still the
            # cap we want. Once a mutation has landed, "skip" means this phase
            # silently keeps the PREVIOUS rung's cap, and the cell would be
            # scored at a limit it never received.
            if self._last_applied is not None and self._last_applied != (
                new_vram_bytes,
                new_sm_percent,
            ):
                raise RuntimeError(
                    f"HAMi initialized_flag stayed 0 after {INIT_WAIT_TIMEOUT_S:.1f}s "
                    f"for container '{self._target_metadata.get('container_name')}', so "
                    f"this phase's cap could not be applied. The container is still at "
                    f"limit={self._last_applied[0]} bytes / sm={self._last_applied[1]}%, "
                    f"not the requested limit={new_vram_bytes} / sm={new_sm_percent}. "
                    f"Refusing to score a phase at a cap it never received."
                )
            logger.warning(
                "HAMi initialized_flag stayed 0 after %.1fs; skipping mutation "
                "for this phase. Container may not have triggered CUDA init yet.",
                INIT_WAIT_TIMEOUT_S,
            )
            return

        if not self._version_checked:
            self._verify_version()
            self._version_checked = True

        try:
            with self._cache_host_path.open("r+b") as f:
                f.seek(LIMIT_OFFSET)
                f.write(struct.pack("<Q", new_vram_bytes))
                f.seek(SM_LIMIT_OFFSET)
                f.write(struct.pack("<Q", new_sm_percent))
                # Read back inside the same handle, before anything else can
                # touch the region. Mirrors _verify_cpu_applied: a write that
                # reports success is not evidence the value landed, and an
                # unverified GPU cap is the one axis that was trusted on faith
                # while CPU and memory were both checked.
                f.flush()
                f.seek(LIMIT_OFFSET)
                got_limit = struct.unpack("<Q", f.read(8))[0]
                f.seek(SM_LIMIT_OFFSET)
                got_sm = struct.unpack("<Q", f.read(8))[0]
        except OSError as exc:
            raise RuntimeError(
                f"HAMi runtime mutation failed for container "
                f"'{self._target_metadata.get('container_name')}': {exc}. The cap was "
                f"not applied, so this phase would be scored at a limit it never "
                f"received. Refusing to continue."
            ) from exc

        if (got_limit, got_sm) != (new_vram_bytes, new_sm_percent):
            raise RuntimeError(
                f"HAMi cap did not take effect for container "
                f"'{self._target_metadata.get('container_name')}': wrote "
                f"limit={new_vram_bytes} bytes / sm={new_sm_percent}% but the shared "
                f"region reads back limit={got_limit} / sm={got_sm}. Either the write "
                f"was a silent no-op or LIMIT_OFFSET/SM_LIMIT_OFFSET no longer match "
                f"this libvgpu.so (see _verify_version). Refusing to score a cell at a "
                f"cap it never received."
            )

        self._last_applied = (new_vram_bytes, new_sm_percent)
        logger.info(
            "HAMi runtime mutation: container '%s' limit[0]=%s bytes (%.2f GB), "
            "sm_limit[0]=%s%% (read back OK)",
            self._target_metadata.get("container_name"),
            new_vram_bytes,
            new_vram_bytes / (1024 ** 3),
            new_sm_percent,
        )

    def release(self) -> None:
        if self._cache_host_path is None or not self._cache_host_path.exists():
            return
        try:
            with self._cache_host_path.open("r+b") as f:
                f.seek(LIMIT_OFFSET)
                f.write(struct.pack("<Q", 0))
                f.seek(SM_LIMIT_OFFSET)
                f.write(struct.pack("<Q", 0))
        except OSError as exc:
            logger.warning(
                "HAMi cap release (zero-out) failed for '%s': %s",
                self._cache_host_path,
                exc,
            )

    def cleanup(self) -> None:
        self._remove_legacy_shared_cache()
        if self._cache_host_path is not None:
            try:
                self._cache_host_path.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                logger.warning(
                    "Failed to remove per-container HAMi cache '%s': %s",
                    self._cache_host_path,
                    exc,
                )
        self._cache_host_path = None
        self._process = None
        self._target_kind = None
        self._target_metadata = {}
        self._init_observed = False
        self._version_checked = False
        self._last_applied = None

    # ----- internal helpers -----

    def _wait_for_init(self) -> bool:
        """Poll initialized_flag until non-zero or timeout. Returns True if observed."""
        if self._cache_host_path is None:
            return False
        deadline = time.monotonic() + INIT_WAIT_TIMEOUT_S
        while time.monotonic() < deadline:
            try:
                with self._cache_host_path.open("rb") as f:
                    f.seek(INITIALIZED_FLAG_OFFSET)
                    flag = struct.unpack("<i", f.read(4))[0]
                if flag != 0:
                    self._init_observed = True
                    return True
            except OSError:
                pass
            time.sleep(INIT_POLL_INTERVAL_S)
        return False

    def _verify_version(self) -> None:
        """Read major/minor version from the cache file; warn on mismatch."""
        if self._cache_host_path is None:
            return
        try:
            with self._cache_host_path.open("rb") as f:
                f.seek(MAJOR_VERSION_OFFSET)
                major, minor = struct.unpack("<II", f.read(8))
        except OSError:
            return
        if major != EXPECTED_MAJOR_VERSION or minor != EXPECTED_MINOR_VERSION:
            logger.warning(
                "HAMi cache version mismatch: file reports %d.%d, expected %d.%d. "
                "shared_region_t layout may have shifted; runtime mutation may "
                "corrupt state. Re-run the offsetof probe in "
                "this project's HAMi runtime-mutation notes against the "
                "current libvgpu.so and update LIMIT_OFFSET / SM_LIMIT_OFFSET.",
                major, minor, EXPECTED_MAJOR_VERSION, EXPECTED_MINOR_VERSION,
            )

    def _remove_legacy_shared_cache(self) -> None:
        """Best-effort removal of the legacy shared HAMi cache file.

        Pre-runtime-mutation, the controller wrote/read a single global
        ``/tmp/cudevshr.cache`` file. With per-container bind-mounted
        cache files, the legacy global path is irrelevant — but we still
        clean it up to keep the host state tidy.
        """
        cache_path = Path(HAMI_SHARED_CACHE)
        try:
            if cache_path.is_file() or cache_path.is_symlink():
                cache_path.unlink()
            elif cache_path.is_dir():
                shutil.rmtree(cache_path, ignore_errors=True)
        except FileNotFoundError:
            return
        except Exception as exc:
            logger.warning(
                "Failed to remove legacy HAMi shared cache at '%s': %s",
                cache_path,
                exc,
            )
