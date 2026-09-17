"""Runtime-stress controllers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import json
import logging
import subprocess
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from .models import RuntimeStressControls


class ResourceController(ABC):
    """Interface for a runtime-stress controller."""

    @abstractmethod
    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Prepare controller state for one running process."""

    @abstractmethod
    def apply(self, controls: RuntimeStressControls) -> None:
        """Apply the current phase controls."""

    @abstractmethod
    def release(self) -> None:
        """Release any active controls."""

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up controller resources."""


class CpuQuotaController(ResourceController):
    """Apply CPU quota limits using cgroup v2 cpu.max."""

    def __init__(self, period_us: int = 100_000):
        self.period_us = period_us
        self._process = None
        self._cgroup_path: Path | None = None
        self._cpu_max_path: Path | None = None
        self._original_cpu_max: str | None = None

    def prepare(
        self,
        process,
        *,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if target_kind != "host_process_group":
            raise RuntimeError(
                f"CpuQuotaController only supports host_process_group targets, got '{target_kind}'."
            )

        self._process = process
        cgroup_root = self._detect_cgroup_v2_mount()
        cgroup_path = self._resolve_process_cgroup_path(process.pid, cgroup_root)
        cpu_max_path = cgroup_path / "cpu.max"
        controllers_path = cgroup_path / "cgroup.controllers"

        if not cgroup_path.exists() or not cgroup_path.is_dir():
            raise RuntimeError(f"Resolved cgroup v2 path is not a directory: {cgroup_path}")

        if not cpu_max_path.exists() or not cpu_max_path.is_file():
            delegated_controllers = ""
            if controllers_path.exists():
                delegated_controllers = controllers_path.read_text(encoding="utf-8").strip()

            if "cpu" not in delegated_controllers.split():
                raise RuntimeError(
                    "CPU runtime stress is unavailable in the target cgroup because the cgroup v2 "
                    f"CPU controller is not delegated there.\n"
                    f"Target cgroup: {cgroup_path}\n"
                    "On this host, run the evaluation in a delegated system scope, for example:\n"
                    "  sudo systemd-run --scope -p Delegate=yes -p CPUAccounting=yes bash\n"
                    "Then rerun the evaluation from that shell."
                )

            raise RuntimeError(
                f"CPU runtime stress requires a writable cgroup v2 cpu.max file, but none was found at {cpu_max_path}."
            )

        self._cgroup_path = cgroup_path
        self._cpu_max_path = cpu_max_path
        try:
            self._original_cpu_max = cpu_max_path.read_text(encoding="utf-8")
            self.release()
        except PermissionError as exc:
            raise RuntimeError(
                "CPU runtime stress cannot write to the target cgroup. "
                "Run the evaluation inside a delegated user scope, for example:\n"
                "  systemd-run --user --scope -p Delegate=yes bash\n"
                "Then rerun the evaluation from that shell."
            ) from exc

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._cpu_max_path is None:
            raise RuntimeError("CPU controller must be prepared before apply().")

        cpu = controls.cpu
        if cpu is None or cpu.max_cores is None:
            self.release()
            return

        quota_us = max(1, int(round(cpu.max_cores * self.period_us)))
        self._cpu_max_path.write_text(
            f"{quota_us} {self.period_us}\n",
            encoding="utf-8",
        )

    def release(self) -> None:
        if self._cpu_max_path is None:
            return
        restored_value = self._original_cpu_max or f"max {self.period_us}\n"
        self._cpu_max_path.write_text(restored_value, encoding="utf-8")

    def cleanup(self) -> None:
        if self._cpu_max_path is None:
            return

        try:
            self.release()
        except Exception:
            pass

        self._cgroup_path = None
        self._cpu_max_path = None
        self._original_cpu_max = None
        self._process = None

    def _detect_cgroup_v2_mount(self) -> Path:
        """Return the active cgroup v2 mount path for this host."""
        preferred = [
            Path("/sys/fs/cgroup"),
            Path("/sys/fs/cgroup/unified"),
        ]
        for candidate in preferred:
            if (candidate / "cgroup.controllers").exists():
                return candidate

        with open("/proc/self/mountinfo", "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                fields = line.strip().split()
                if "-" not in fields:
                    continue
                separator_index = fields.index("-")
                if separator_index + 2 >= len(fields):
                    continue
                if fields[separator_index + 1] != "cgroup2":
                    continue

                mount_path = Path(fields[4])
                if (mount_path / "cgroup.controllers").exists():
                    return mount_path

        raise RuntimeError("CPU runtime stress requires a mounted cgroup v2 hierarchy.")

    def _resolve_process_cgroup_path(self, pid: int, cgroup_root: Path) -> Path:
        """Resolve the cgroup v2 directory for a running process."""
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as file_handle:
            for line in file_handle:
                _, _, path = line.rstrip("\n").partition("::")
                if not path:
                    continue
                if path == "/":
                    return cgroup_root
                return cgroup_root / path.lstrip("/")

        raise RuntimeError(f"Could not resolve cgroup v2 path for pid {pid}.")


class DockerCpuController(ResourceController):
    """Apply CPU quota limits to a running Docker container."""

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
        if target_kind != "docker_container":
            raise RuntimeError(
                f"DockerCpuController only supports docker_container targets, got '{target_kind}'."
            )

        if not isinstance(target_metadata, dict):
            raise RuntimeError("Docker CPU runtime stress requires target metadata with container_name.")

        container_name = target_metadata.get("container_name")
        if not isinstance(container_name, str) or not container_name.strip():
            raise RuntimeError("Docker CPU runtime stress requires a non-empty container_name.")

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
            f"Docker CPU runtime stress could not attach to container '{self._container_name}'."
        )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._container_name is None:
            raise RuntimeError("Docker CPU controller must be prepared before apply().")

        cpu = controls.cpu
        if cpu is None or cpu.max_cores is None:
            self.release()
            return

        self._docker_update("--cpus", f"{float(cpu.max_cores):g}")

    def release(self) -> None:
        if self._container_name is None:
            return

        restore_cpus = 0.0
        if self._original_nano_cpus and self._original_nano_cpus > 0:
            restore_cpus = self._original_nano_cpus / 1_000_000_000

        self._docker_update("--cpus", f"{restore_cpus:g}")

    def cleanup(self) -> None:
        self._process = None
        self._container_name = None
        self._original_nano_cpus = None

    def _inspect_container(self, container_name: str) -> Optional[Dict[str, Any]]:
        result = subprocess.run(
            ["docker", "inspect", container_name],
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

    def _docker_update(self, option: str, value: str) -> None:
        if self._container_name is None:
            raise RuntimeError("Docker CPU controller is not attached to a container.")

        result = subprocess.run(
            ["docker", "update", option, value, self._container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return

        stderr_text = (result.stderr or "").strip()
        if "No such container" in stderr_text:
            return

        raise RuntimeError(
            f"Docker CPU runtime stress failed for container '{self._container_name}': {stderr_text or result.stdout.strip()}"
        )


class DockerMemoryController(ResourceController):
    """Apply memory limits to a running Docker container."""

    def __init__(self, attach_timeout_s: float = 15.0):
        self.attach_timeout_s = attach_timeout_s
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
        if target_kind != "docker_container":
            raise RuntimeError(
                f"DockerMemoryController only supports docker_container targets, got '{target_kind}'."
            )

        if not isinstance(target_metadata, dict):
            raise RuntimeError("Docker memory runtime stress requires target metadata with container_name.")

        container_name = target_metadata.get("container_name")
        if not isinstance(container_name, str) or not container_name.strip():
            raise RuntimeError("Docker memory runtime stress requires a non-empty container_name.")

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
            f"Docker memory runtime stress could not attach to container '{self._container_name}'."
        )

    def apply(self, controls: RuntimeStressControls) -> None:
        if self._container_name is None:
            raise RuntimeError("Docker memory controller must be prepared before apply().")

        memory = controls.memory
        if memory is None or memory.max_mb is None:
            self.release()
            return

        limit = f"{memory.max_mb}m"
        self._docker_update("--memory", limit, "--memory-swap", limit)

    def release(self) -> None:
        if self._container_name is None:
            return

        if self._original_memory_bytes and self._original_memory_bytes > 0:
            mem = str(self._original_memory_bytes)
            swap = str(self._original_memory_swap_bytes) if self._original_memory_swap_bytes else mem
            self._docker_update("--memory-swap", swap, "--memory", mem)
        else:
            self._docker_update("--memory-swap", "-1", "--memory", "0")

    def cleanup(self) -> None:
        self._process = None
        self._container_name = None
        self._original_memory_bytes = None
        self._original_memory_swap_bytes = None

    def _inspect_container(self, container_name: str) -> Optional[Dict[str, Any]]:
        result = subprocess.run(
            ["docker", "inspect", container_name],
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

    def _docker_update(self, *args: str) -> None:
        if self._container_name is None:
            raise RuntimeError("Docker memory controller is not attached to a container.")

        result = subprocess.run(
            ["docker", "update", *args, self._container_name],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 0:
            return

        stderr_text = (result.stderr or "").strip()
        if "No such container" in stderr_text:
            return

        if "unable to set memory limit" in stderr_text:
            logger.warning(
                "Memory limit rejected by kernel (current usage exceeds target) "
                "for container '%s': %s",
                self._container_name,
                stderr_text,
            )
            return

        raise RuntimeError(
            f"Docker memory runtime stress failed for container '{self._container_name}': {stderr_text or result.stdout.strip()}"
        )
