"""Runtime-stress controllers for rootless Podman containers."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .controllers import ResourceController
from .models import IoControl, RuntimeStressControls

logger = logging.getLogger(__name__)

CGROUP_V2_CONTROLLERS_PATH = Path("/sys/fs/cgroup/cgroup.controllers")
NVME_INEFFECTIVE_SCHEDULERS = {"none", "mq-deadline"}

# Sentinels for "effectively unlimited" used by release() paths.
#
# `podman update --cpus 0` / `--memory 0` is documented to mean "no limit"
# (matching `podman run`), but on Podman 5.8.2 calling update with 0 after a
# real cap has been set is a silent no-op: neither cgroup nor HostConfig
# changes. See the related upstream issue
# https://github.com/containers/podman/issues/17880 ("podman update doesn't
# appear to be able to change memory limits") for the same family of bug.
#
# Workaround: pass values so large they're effectively unlimited on any real
# hardware. Podman accepts and applies them; the kernel page-aligns memory
# down (cosmetic).
_RELEASE_CPUS_SENTINEL = 1_000_000           # 1M cores >> any hardware
_RELEASE_MEMORY_BYTES_SENTINEL = 1 << 53     # ~9 PB >> any single-host RAM


class PodmanCpuController(ResourceController):
    """Apply CPU quota limits to a running rootless Podman container."""

    def __init__(self, attach_timeout_s: float = 15.0):
        self.attach_timeout_s = attach_timeout_s
        self._process = None
        self._container_name: Optional[str] = None
        self._original_nano_cpus: Optional[int] = None

    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target_kind != "podman_container":
            raise RuntimeError(
                f"PodmanCpuController only supports podman_container targets, got '{target_kind}'."
            )

        if not isinstance(target_metadata, dict):
            raise RuntimeError("Podman CPU runtime stress requires target metadata with container_name.")

        container_name = target_metadata.get("container_name")
        if not isinstance(container_name, str) or not container_name.strip():
            raise RuntimeError("Podman CPU runtime stress requires a non-empty container_name.")

        self._process = process
        self._container_name = container_name.strip()

        deadline = time.monotonic() + self.attach_timeout_s
        while time.monotonic() < deadline:
            inspect_data = self._inspect_container(self._container_name)
            if inspect_data is not None:
                host_config = inspect_data.get("HostConfig", {})
                self._original_nano_cpus = int(host_config.get("NanoCpus") or 0)
                self.release()
                return

            if process.poll() is not None:
                break
            time.sleep(0.1)

        raise RuntimeError(
            f"Podman CPU runtime stress could not attach to container '{self._container_name}'."
        )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._container_name is None:
            raise RuntimeError("Podman CPU controller must be prepared before apply().")

        cpu = controls.cpu
        if cpu is None or cpu.max_cores is None:
            self.release()
            return

        if self._podman_update("--cpus", f"{float(cpu.max_cores):g}"):
            self._verify_cpu_applied(float(cpu.max_cores))

    def _verify_cpu_applied(self, requested_cores: float) -> None:
        """Fail loud if the cap did not actually take effect.

        ``podman update`` can return success while silently applying nothing
        (the documented ``--cpus 0`` / issue 17880 footgun). Trusting the exit
        code alone risks scoring an unstressed cell as CPU-stressed, which would
        silently corrupt a stress sweep. Read the applied limit back and raise
        if it is missing or wrong.
        """
        inspect = self._inspect_container(self._container_name)
        if inspect is None:
            return  # container exited; the run is over, nothing to enforce
        nano = int((inspect.get("HostConfig") or {}).get("NanoCpus") or 0)
        expected = int(round(requested_cores * 1_000_000_000))
        tol = max(1_000_000, int(expected * 0.02))
        if nano <= 0 or abs(nano - expected) > tol:
            raise RuntimeError(
                f"Podman CPU cap did not take effect for '{self._container_name}': "
                f"requested {requested_cores:g} cores (NanoCpus={expected}) but the "
                f"container reports NanoCpus={nano}. 'podman update --cpus' returned "
                f"success yet the cap was a silent no-op (podman issue-17880 family). "
                f"Refusing to score an unstressed cell as CPU-stressed."
            )

    def release(self) -> None:
        if self._container_name is None:
            return

        if self._original_nano_cpus and self._original_nano_cpus > 0:
            restore_cpus = self._original_nano_cpus / 1_000_000_000
            self._podman_update("--cpus", f"{restore_cpus:g}")
            self._verify_cpu_released(self._original_nano_cpus)
            return

        # Container was originally unlimited. `--cpus 0` is documented to
        # mean "no limit" but is a silent no-op once a real cap has been
        # applied (Podman 5.8.2). Pass an effectively-unlimited sentinel.
        self._podman_update("--cpus", f"{_RELEASE_CPUS_SENTINEL}")
        self._verify_cpu_released(None)

    def _verify_cpu_released(self, expected_nano: Optional[int]) -> None:
        """Fail loud if the cap is still in place after releasing it.

        The mirror of `_verify_cpu_applied`, and needed for the same reason: the
        release goes through the SAME ``podman update`` that is documented to
        return success while applying nothing. An unverified release fails in the
        more misleading direction. A cap that silently outlives its phase leaves
        the next phase still squeezed, so a system that was never actually let go
        gets scored as having failed to recover -- a false failure invented by the
        harness, which is exactly what the recovery experiment must not produce.

        A container that has exited is not an error: there is no cap left to
        outlive it, and nothing downstream to contaminate.
        """
        inspect = self._inspect_container(self._container_name)
        if inspect is None:
            return
        nano = int((inspect.get("HostConfig") or {}).get("NanoCpus") or 0)

        if expected_nano:
            # Restored to a real pre-existing cap: it must read back as that cap.
            tol = max(1_000_000, int(expected_nano * 0.02))
            if abs(nano - expected_nano) > tol:
                raise RuntimeError(
                    f"Podman CPU cap was not restored for "
                    f"'{self._container_name}': expected NanoCpus="
                    f"{expected_nano} (the container's original limit) but it "
                    f"reports NanoCpus={nano}. 'podman update --cpus' returned "
                    f"success yet the restore was a silent no-op. Refusing to "
                    f"score a still-capped phase as released."
                )
            return

        # Originally unlimited. 0 means no limit; otherwise the sentinel must be
        # in place. Anything smaller is a cap that survived its phase.
        sentinel_nano = int(_RELEASE_CPUS_SENTINEL) * 1_000_000_000
        if nano != 0 and nano < sentinel_nano:
            raise RuntimeError(
                f"Podman CPU cap outlived its phase for "
                f"'{self._container_name}': expected no limit (NanoCpus=0 or the "
                f"{_RELEASE_CPUS_SENTINEL}-core release sentinel) but it reports "
                f"NanoCpus={nano}. 'podman update --cpus' returned success yet "
                f"the release was a silent no-op (the documented '--cpus 0' "
                f"footgun). Refusing to score a still-capped phase as released."
            )

    def cleanup(self) -> None:
        self._process = None
        self._container_name = None
        self._original_nano_cpus = None

    def _inspect_container(self, container_name: str) -> Optional[Dict[str, Any]]:
        result = subprocess.run(
            ["podman", "inspect", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        payload = json.loads(result.stdout)
        if not payload:
            return None
        return payload[0]

    def _podman_update(self, option: str, value: str) -> bool:
        """Run ``podman update``. Returns True when it reported success (the
        cap may then be verified), False when the call was a tolerated non-apply
        (container gone), and raises on a genuine error."""
        if self._container_name is None:
            raise RuntimeError("Podman CPU controller is not attached to a container.")

        result = subprocess.run(
            ["podman", "update", option, value, self._container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return True

        stderr_text = (result.stderr or "").strip()
        if "no such container" in stderr_text.lower():
            return False

        raise RuntimeError(
            f"Podman CPU runtime stress failed for container '{self._container_name}': {stderr_text or result.stdout.strip()}"
        )


class PodmanMemoryController(ResourceController):
    """Apply memory limits to a running rootless Podman container."""

    def __init__(self, attach_timeout_s: float = 15.0,
                 preapplied_max_mb: Optional[int] = None):
        """``preapplied_max_mb`` names a cap already set by ``podman run
        --memory`` at launch, so ``apply()`` can skip the ``podman update``
        for it. That update is the call that hung on okvis2x while lowering a
        limit. The skip never weakens the check: the cap is still read back
        from the container and still raises if it is not actually in place.

        A launch-time cap is in force for the WHOLE run, warmup included,
        which a mid-run cap is not. The orchestrator only offers one when
        every phase that declares a memory cap declares the same value, so
        there is no per-phase step for the early application to erase."""
        self.attach_timeout_s = attach_timeout_s
        self.preapplied_max_mb = preapplied_max_mb
        self._process = None
        self._container_name: Optional[str] = None
        self._original_memory_bytes: Optional[int] = None
        self._original_memory_swap_bytes: Optional[int] = None

    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target_kind != "podman_container":
            raise RuntimeError(
                f"PodmanMemoryController only supports podman_container targets, got '{target_kind}'."
            )

        if not isinstance(target_metadata, dict):
            raise RuntimeError("Podman memory runtime stress requires target metadata with container_name.")

        container_name = target_metadata.get("container_name")
        if not isinstance(container_name, str) or not container_name.strip():
            raise RuntimeError("Podman memory runtime stress requires a non-empty container_name.")

        self._process = process
        self._container_name = container_name.strip()

        deadline = time.monotonic() + self.attach_timeout_s
        while time.monotonic() < deadline:
            inspect_data = self._inspect_container(self._container_name)
            if inspect_data is not None:
                host_config = inspect_data.get("HostConfig", {})
                self._original_memory_bytes = int(host_config.get("Memory") or 0)
                self._original_memory_swap_bytes = int(host_config.get("MemorySwap") or 0)
                self.release()
                return

            if process.poll() is not None:
                break
            time.sleep(0.1)

        raise RuntimeError(
            f"Podman memory runtime stress could not attach to container '{self._container_name}'."
        )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._container_name is None:
            raise RuntimeError("Podman memory controller must be prepared before apply().")

        memory = controls.memory
        if memory is None or memory.max_mb is None:
            self.release()
            return

        if (self.preapplied_max_mb is not None
                and int(memory.max_mb) == int(self.preapplied_max_mb)):
            # Already capped at launch. Verify rather than trust it.
            self._verify_memory_applied(int(memory.max_mb))
            return

        limit = f"{memory.max_mb}m"
        if self._podman_update("--memory", limit):
            self._verify_memory_applied(int(memory.max_mb))

    def _verify_memory_applied(self, requested_mb: int) -> None:
        """Fail loud if the memory cap did not take effect (issue 17880 was
        reported specifically against ``podman update --memory``). Read the
        applied limit back and raise on a silent no-op, so a stressed cell can
        never be scored as unstressed."""
        inspect = self._inspect_container(self._container_name)
        if inspect is None:
            return  # container exited; nothing to enforce
        mem = int((inspect.get("HostConfig") or {}).get("Memory") or 0)
        expected = requested_mb * 1024 * 1024
        tol = max(4096, int(expected * 0.02))  # kernel may page-align; allow slack
        if mem <= 0 or abs(mem - expected) > tol:
            raise RuntimeError(
                f"Podman memory cap did not take effect for '{self._container_name}': "
                f"requested {requested_mb} MB ({expected} bytes) but the container "
                f"reports Memory={mem}. 'podman update --memory' returned success yet "
                f"the cap was a silent no-op (podman issue-17880). Refusing to score "
                f"an unstressed cell as memory-stressed."
            )

    def release(self) -> None:
        if self._container_name is None:
            return

        if self.preapplied_max_mb is not None:
            # A launch-time cap is a property of the container, not of a phase.
            # Lifting it between phases would both undo the treatment and route
            # through the very ``podman update`` this path exists to avoid.
            return

        if self._original_memory_bytes and self._original_memory_bytes > 0:
            self._podman_update("--memory", str(self._original_memory_bytes))
            return

        # Container was originally unlimited. Same `--memory 0` no-op as the
        # CPU controller; pass an effectively-unlimited sentinel.
        self._podman_update("--memory", str(_RELEASE_MEMORY_BYTES_SENTINEL))

    def cleanup(self) -> None:
        self._process = None
        self._container_name = None
        self._original_memory_bytes = None
        self._original_memory_swap_bytes = None

    def _inspect_container(self, container_name: str) -> Optional[Dict[str, Any]]:
        result = subprocess.run(
            ["podman", "inspect", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        payload = json.loads(result.stdout)
        if not payload:
            return None
        return payload[0]

    def _podman_update(self, *args: str) -> bool:
        if self._container_name is None:
            raise RuntimeError("Podman memory controller is not attached to a container.")

        result = subprocess.run(
            ["podman", "update", *args, self._container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return True

        stderr_text = (result.stderr or "").strip()
        if "no such container" in stderr_text.lower():
            return False

        if "unable to set memory limit" in stderr_text.lower():
            logger.warning(
                "Memory limit rejected by kernel (current usage exceeds target) "
                "for container '%s': %s",
                self._container_name,
                stderr_text,
            )
            return False

        raise RuntimeError(
            f"Podman memory runtime stress failed for container '{self._container_name}': {stderr_text or result.stdout.strip()}"
        )


@dataclass
class _IoDevice:
    """One whole-disk block device that the controller will throttle."""

    path: str            # e.g. /dev/sda
    major: int
    minor: int
    scheduler: Optional[str] = None  # currently active IO scheduler, if known
    pseudo: bool = False             # loop/dm/overlay-backed → throttle is best-effort


class PodmanIoController(ResourceController):
    """Apply block-IO bandwidth and IOPS limits to a running rootless Podman container.

    The kernel cgroup v2 ``io.max`` interface only accepts whole-disk devices,
    so paths from ``target_metadata['io_target_paths']`` are resolved via
    ``findmnt`` and then walked via ``lsblk -no PKNAME`` to the underlying
    physical disk before being throttled. Caps are applied with
    ``podman update --device-{read,write}-{bps,iops}`` and released by writing
    ``<maj>:<min> rbps=max wbps=max riops=max wiops=max`` directly to the
    container's ``io.max`` (``podman update`` rejects ``:0`` with "out of range").
    """

    def __init__(self, attach_timeout_s: float = 15.0):
        self.attach_timeout_s = attach_timeout_s
        self._process = None
        self._container_name: Optional[str] = None
        self._target_devices: List[_IoDevice] = []
        self._cgroup_path: Optional[str] = None
        self._original_read_bps: Dict[str, int] = {}
        self._original_write_bps: Dict[str, int] = {}
        self._original_read_iops: Dict[str, int] = {}
        self._original_write_iops: Dict[str, int] = {}
        self._has_active_throttle: bool = False
        self._iops_warning_emitted: bool = False

    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target_kind != "podman_container":
            raise RuntimeError(
                f"PodmanIoController only supports podman_container targets, got '{target_kind}'."
            )

        if not isinstance(target_metadata, dict):
            raise RuntimeError("Podman IO runtime stress requires target metadata with container_name.")

        container_name = target_metadata.get("container_name")
        if not isinstance(container_name, str) or not container_name.strip():
            raise RuntimeError("Podman IO runtime stress requires a non-empty container_name.")

        _verify_cgroup_io_delegated()

        self._process = process
        self._container_name = container_name.strip()

        target_paths = list(target_metadata.get("io_target_paths") or [])
        if not target_paths:
            raise RuntimeError(
                "Podman IO runtime stress requires target metadata with io_target_paths "
                "(host paths backing the container's bind mounts)."
            )

        self._target_devices = _resolve_target_devices(target_paths)
        if not self._target_devices:
            raise RuntimeError(
                f"Podman IO runtime stress could not resolve any block device from "
                f"io_target_paths={target_paths!r}."
            )

        deadline = time.monotonic() + self.attach_timeout_s
        while time.monotonic() < deadline:
            inspect_data = self._inspect_container(self._container_name)
            if inspect_data is not None:
                host_config = inspect_data.get("HostConfig", {}) or {}
                self._original_read_bps = _bps_dict(host_config.get("BlkioDeviceReadBps"))
                self._original_write_bps = _bps_dict(host_config.get("BlkioDeviceWriteBps"))
                self._original_read_iops = _bps_dict(host_config.get("BlkioDeviceReadIOps"))
                self._original_write_iops = _bps_dict(host_config.get("BlkioDeviceWriteIOps"))

                state = inspect_data.get("State", {}) or {}
                self._cgroup_path = state.get("CgroupPath") or None
                return

            if process.poll() is not None:
                break
            time.sleep(0.1)

        raise RuntimeError(
            f"Podman IO runtime stress could not attach to container '{self._container_name}'."
        )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._container_name is None:
            raise RuntimeError("Podman IO controller must be prepared before apply().")

        io = controls.io
        if io is None or _io_all_none(io):
            self.release()
            return

        self._maybe_warn_iops_on_nvme(io)

        update_args: List[str] = []
        for device in self._target_devices:
            if io.read_bps is not None:
                update_args.extend(["--device-read-bps", f"{device.path}:{int(io.read_bps)}"])
            if io.write_bps is not None:
                update_args.extend(["--device-write-bps", f"{device.path}:{int(io.write_bps)}"])
            if io.read_iops is not None:
                update_args.extend(["--device-read-iops", f"{device.path}:{int(io.read_iops)}"])
            if io.write_iops is not None:
                update_args.extend(["--device-write-iops", f"{device.path}:{int(io.write_iops)}"])

        if not update_args:
            return

        self._podman_update(update_args)
        self._has_active_throttle = True
        self._enforce_io_max(io)

    def _enforce_io_max(self, io: IoControl) -> None:
        """Write io.max directly, because `podman update` does not.

        MEASURED, not assumed: under rootless podman on this host,
        `podman update --device-read-bps /dev/sda:100000000` exits 0 and records
        `BlkioDeviceReadBps: [{/dev/sda 100000000}]` in inspect, while the
        container's `io.max` stays EMPTY. The cap is accepted, reported as
        applied, and never enforced -- the same silent-no-op family as the CPU
        and memory footguns this file already guards against, except nothing was
        checking this one. Every IO cell measured before this fix was unthrottled
        no matter what its config said.

        A direct write works: `echo '8:0 rbps=100000000' > io.max` takes effect
        immediately, which is why `release()` has always written the cgroup
        directly rather than going through podman. Apply now uses the same path,
        and then reads it back, so an unenforceable cap fails the run instead of
        quietly producing an uncapped result labelled as capped.
        """
        cgroup_io_max = self._cgroup_io_max_path()
        logger.info("IO enforce: cgroup io.max path resolved to %s", cgroup_io_max)
        if cgroup_io_max is None:
            raise RuntimeError(
                "Podman IO controller cannot resolve the container cgroup, so an "
                "IO cap cannot be enforced. `podman update --device-*-bps` reports "
                "success without writing io.max, so continuing would score an "
                "UNTHROTTLED run as IO-limited. Refusing."
            )

        def tok(value):
            return str(int(value)) if value is not None else "max"

        for device in self._target_devices:
            line = (
                f"{device.major}:{device.minor} "
                f"rbps={tok(io.read_bps)} wbps={tok(io.write_bps)} "
                f"riops={tok(io.read_iops)} wiops={tok(io.write_iops)}"
            )
            try:
                cgroup_io_max.write_text(line)
            except (FileNotFoundError, PermissionError, OSError) as exc:
                raise RuntimeError(
                    f"Could not write io.max for {device.path} ({exc}). An IO cap "
                    f"that cannot be installed must not be scored as applied."
                ) from exc

            # Read back: the kernel accepts the write but only keeps limits it can
            # enforce, so presence of our device line is the proof.
            try:
                current = cgroup_io_max.read_text()
            except OSError:
                current = ""
            prefix = f"{device.major}:{device.minor} "
            if not any(ln.startswith(prefix) for ln in current.splitlines()):
                raise RuntimeError(
                    f"io.max carries no entry for {device.path} "
                    f"({device.major}:{device.minor}) after writing it. The IO cap "
                    f"did not take effect. Refusing to score an unthrottled run as "
                    f"IO-limited."
                )

    def release(self) -> None:
        if self._container_name is None:
            return

        if not self._has_active_throttle and not (
            self._original_read_bps
            or self._original_write_bps
            or self._original_read_iops
            or self._original_write_iops
        ):
            return

        # Restore originals for any device we touched. `podman update --device-X-bps :0`
        # is rejected by the kernel ("Numerical result out of range"), so unset is done
        # by writing rbps=max etc. directly to the container's cgroup io.max.
        cgroup_io_max = self._cgroup_io_max_path()
        for device in self._target_devices:
            tokens = ["max", "max", "max", "max"]  # rbps, wbps, riops, wiops

            for axis_index, axis_originals in enumerate(
                (self._original_read_bps, self._original_write_bps,
                 self._original_read_iops, self._original_write_iops)
            ):
                original_rate = axis_originals.get(device.path)
                if original_rate and original_rate > 0:
                    tokens[axis_index] = str(int(original_rate))

            line = (
                f"{device.major}:{device.minor} "
                f"rbps={tokens[0]} wbps={tokens[1]} riops={tokens[2]} wiops={tokens[3]}"
            )

            if cgroup_io_max is None:
                logger.warning(
                    "Podman IO controller cannot resolve container cgroup; skipping unset for %s",
                    device.path,
                )
                continue

            try:
                cgroup_io_max.write_text(line)
            except (FileNotFoundError, PermissionError) as exc:
                logger.warning(
                    "Podman IO controller could not write '%s' to %s: %s",
                    line, cgroup_io_max, exc,
                )

        self._has_active_throttle = False

    def cleanup(self) -> None:
        self._process = None
        self._container_name = None
        self._target_devices = []
        self._cgroup_path = None
        self._original_read_bps = {}
        self._original_write_bps = {}
        self._original_read_iops = {}
        self._original_write_iops = {}
        self._has_active_throttle = False
        self._iops_warning_emitted = False

    # ----- introspection helpers -----

    def target_device_summaries(self) -> List[Dict[str, Optional[str]]]:
        """Public summary of throttle target devices for `stress_summary.json`."""
        return [
            {
                "path": device.path,
                "major_minor": f"{device.major}:{device.minor}",
                "scheduler": device.scheduler,
                "pseudo": device.pseudo,
            }
            for device in self._target_devices
        ]

    # ----- internal helpers -----

    def _inspect_container(self, container_name: str) -> Optional[Dict[str, Any]]:
        result = subprocess.run(
            ["podman", "inspect", container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return None

        payload = json.loads(result.stdout)
        if not payload:
            return None
        return payload[0]

    def _podman_update(self, args: List[str]) -> None:
        if self._container_name is None:
            raise RuntimeError("Podman IO controller is not attached to a container.")

        result = subprocess.run(
            ["podman", "update", *args, self._container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return

        stderr_text = (result.stderr or "").strip()
        if "no such container" in stderr_text.lower():
            return

        raise RuntimeError(
            f"Podman IO runtime stress failed for container '{self._container_name}': "
            f"{stderr_text or result.stdout.strip()}"
        )

    def _cgroup_io_max_path(self) -> Optional[Path]:
        if not self._cgroup_path:
            return None
        cgroup_root = Path("/sys/fs/cgroup")
        relative = self._cgroup_path.lstrip("/")
        # podman places the workload PIDs under <scope>/container; the scope itself
        # has io.max but writes there don't override the leaf. Try the leaf first.
        candidates = [
            cgroup_root / relative / "container" / "io.max",
            cgroup_root / relative / "io.max",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _maybe_warn_iops_on_nvme(self, io: IoControl) -> None:
        if self._iops_warning_emitted:
            return
        if io.read_iops is None and io.write_iops is None:
            return

        for device in self._target_devices:
            scheduler = device.scheduler
            if scheduler in NVME_INEFFECTIVE_SCHEDULERS:
                logger.warning(
                    "IOPS throttling on %s may be ineffective with scheduler '%s'. "
                    "Use 'bfq' for cgroup v2 IOPS enforcement, or expect "
                    "read_iops/write_iops caps to silently no-op.",
                    device.path, scheduler,
                )
                self._iops_warning_emitted = True


def _verify_cgroup_io_delegated() -> None:
    if not CGROUP_V2_CONTROLLERS_PATH.exists():
        raise RuntimeError(
            "IO runtime stress requires cgroup v2 (no /sys/fs/cgroup/cgroup.controllers)."
        )
    controllers = CGROUP_V2_CONTROLLERS_PATH.read_text().split()
    if "io" not in controllers:
        raise RuntimeError(
            "IO runtime stress requires the 'io' controller delegated. "
            f"Found controllers: {' '.join(controllers)}"
        )


def _resolve_target_devices(paths: Iterable[str]) -> List[_IoDevice]:
    seen_paths: set = set()
    devices: List[_IoDevice] = []
    for path in paths:
        if not path:
            continue
        try:
            resolved_path = str(Path(path).resolve())
        except OSError:
            resolved_path = path

        partition_dev = _findmnt_source(resolved_path)
        if not partition_dev:
            logger.warning("Podman IO controller could not resolve mount source for %s", resolved_path)
            continue

        whole_disk = _walk_to_whole_disk(partition_dev)
        if whole_disk in seen_paths:
            continue
        seen_paths.add(whole_disk)

        try:
            stat = os.stat(whole_disk)
        except FileNotFoundError:
            logger.warning("Podman IO controller resolved missing device %s", whole_disk)
            continue

        major = os.major(stat.st_rdev)
        minor = os.minor(stat.st_rdev)
        scheduler = _read_active_scheduler(whole_disk)
        pseudo = _is_pseudo_device(whole_disk)

        if pseudo:
            logger.warning(
                "Podman IO controller targeting a pseudo-device (%s); cgroup IO "
                "throttling may be ineffective for loop/tmpfs/overlay-backed paths.",
                whole_disk,
            )

        devices.append(
            _IoDevice(
                path=whole_disk,
                major=major,
                minor=minor,
                scheduler=scheduler,
                pseudo=pseudo,
            )
        )
    return devices


def _findmnt_source(path: str) -> Optional[str]:
    result = subprocess.run(
        ["findmnt", "-no", "SOURCE", "-T", path],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return None
    source = (result.stdout or "").strip()
    return source or None


def _walk_to_whole_disk(device_path: str) -> str:
    """Map a partition or dm-N device to its underlying whole-disk path.

    Returns ``device_path`` unchanged if it's already a whole disk or if the
    walk fails. Walks at most one PKNAME hop (LVM stacks deeper than that
    are explicitly out of scope for v1 — see plan).
    """
    basename = os.path.basename(device_path)
    if not basename:
        return device_path
    result = subprocess.run(
        ["lsblk", "-no", "PKNAME", device_path],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    if result.returncode != 0:
        return device_path
    parent = (result.stdout or "").strip().splitlines()
    parent_name = parent[0].strip() if parent else ""
    if not parent_name:
        return device_path
    candidate = f"/dev/{parent_name}"
    if Path(candidate).exists():
        return candidate
    return device_path


def _read_active_scheduler(device_path: str) -> Optional[str]:
    basename = os.path.basename(device_path)
    if not basename:
        return None
    scheduler_file = Path(f"/sys/block/{basename}/queue/scheduler")
    if not scheduler_file.exists():
        return None
    try:
        content = scheduler_file.read_text().strip()
    except OSError:
        return None
    for token in content.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return None


def _is_pseudo_device(device_path: str) -> bool:
    basename = os.path.basename(device_path)
    if not basename:
        return False
    return basename.startswith(("loop", "dm-", "ram", "zram"))


def _bps_dict(entries: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    """Convert podman inspect's BlkioDevice* list-of-dicts into {path: rate}."""
    out: Dict[str, int] = {}
    if not entries:
        return out
    for entry in entries:
        path = entry.get("Path")
        rate = entry.get("Rate")
        if isinstance(path, str) and isinstance(rate, int):
            out[path] = rate
    return out


def _io_all_none(io: IoControl) -> bool:
    return (
        io.read_bps is None
        and io.write_bps is None
        and io.read_iops is None
        and io.write_iops is None
    )
