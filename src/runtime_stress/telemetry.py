"""Telemetry readers for runtime-stress sampling.

Split out of ``orchestrator.py``: the orchestrator owns the clock, the phase
index and the container handles, so it keeps the sampling *tick* and stamps
each sample with the phase it belongs to. What it does not need to own is
*how* a counter is read or how a container runtime's output is parsed, which
is everything here.

Three kinds of thing live in this module:

- Pure parsers for ``podman``/``docker stats`` payloads. No I/O, no state.
- Cgroup readers. The CPU counter is read straight from ``cpu.stat`` rather
  than through a per-sample subprocess, because a ``podman stats`` fork can
  take seconds on a saturated machine and the counter it returns is then
  stale by that much.
- Two small stateful readers, ``GpuTelemetryReader`` and
  ``ProcessGroupSampler``, which need to remember something between samples
  (a disabled flag, and the previous per-PID counters respectively).

Everything fails soft: a reader that cannot read returns ``None`` or an empty
mapping rather than raising, so one unavailable counter never kills a run.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# How often to walk the cgroup's process list for the SLAM-vs-load CPU split.
# Coarser than the telemetry tick: the walk is ~1000 small /proc reads under a
# heavy antagonist, and sampling it every tick would perturb the contention
# being measured.
PROCESS_WALK_PERIOD_S = 2.0

_GPU_FIELDS = (
    "gpu_util_percent",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_power_w",
    "gpu_temp_c",
    "gpu_sm_clock_mhz",
)


# --------------------------------------------------------------------------
# Pure parsers for container-runtime stats payloads
# --------------------------------------------------------------------------

def parse_percent_value(value: str) -> Optional[float]:
    """Parse a Docker percentage string like '12.34%'."""
    text = str(value).strip()
    if not text:
        return None
    if text.endswith("%"):
        text = text[:-1]
    try:
        return float(text)
    except ValueError:
        return None


def parse_int_value(value: str) -> Optional[int]:
    """Parse a numeric string to int, returning None on failure."""
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def parse_byte_value(value: str) -> Optional[int]:
    """Parse Docker byte strings with binary or decimal units."""
    text = str(value).strip()
    if not text:
        return None

    match = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]+)?$", text)
    if match is None:
        return None

    magnitude = float(match.group(1))
    unit = (match.group(2) or "B").strip()
    unit_multipliers = {
        "B": 1,
        "kB": 1000,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
        "TB": 1000 ** 4,
        "KiB": 1024,
        "MiB": 1024 ** 2,
        "GiB": 1024 ** 3,
        "TiB": 1024 ** 4,
    }
    multiplier = unit_multipliers.get(unit)
    if multiplier is None:
        return None
    return int(round(magnitude * multiplier))


def parse_cpu_percent(payload: Dict[str, Any]) -> Optional[float]:
    """Extract container CPU percent from a podman/docker stats payload.

    Podman emits ``CPU`` (and ``AvgCPU``) as bare floats already in percent
    units (e.g. ``0.76`` means 0.76 %). Docker emits ``CPUPerc`` as a string
    with a percent suffix (e.g. ``"12.34%"``). Tolerate both.

    NOTE: podman reports ``CPU`` equal to ``AvgCPU`` -- an average over the
    container's entire lifetime, not use since the previous sample. A
    container that idles then burns 2 cores reads 18%, 47%, 67%, 83% ...
    creeping toward 200% and never arriving. Use ``cpu_time_ns`` deltas for
    instantaneous use; this value is kept for continuity and for the run-wide
    average it genuinely reports.
    """
    for key in ("CPU", "AvgCPU"):
        value = payload.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return parse_percent_value(payload.get("CPUPerc", ""))


def parse_memory_usage(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extract (rss_bytes, memory_limit_bytes) from podman/docker stats.

    Podman emits ``MemUsage`` and ``MemLimit`` as bare integers (bytes).
    Docker emits a combined ``MemUsage`` string like ``"56.4MiB / 15.5GiB"``.
    Tolerate both shapes; return ``(None, None)`` if neither is present.
    """
    mem_usage = payload.get("MemUsage")
    mem_limit = payload.get("MemLimit")
    if isinstance(mem_usage, int) and not isinstance(mem_usage, bool):
        return (
            mem_usage,
            mem_limit if isinstance(mem_limit, int) and not isinstance(mem_limit, bool) else None,
        )

    if isinstance(mem_usage, str):
        parts = [part.strip() for part in mem_usage.split("/", maxsplit=1)]
        if len(parts) == 2:
            return parse_byte_value(parts[0]), parse_byte_value(parts[1])

    return None, None


def parse_block_io(payload: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """Extract cumulative read/write bytes from podman/docker stats payload.

    Podman emits ``BlockInput`` / ``BlockOutput`` as bare integers (bytes).
    Docker emits a combined ``BlockIO`` string like ``"12.3MB / 4.5MB"``.
    Tolerate both shapes; return ``(None, None)`` if neither is present.
    """
    block_input = payload.get("BlockInput")
    block_output = payload.get("BlockOutput")
    if isinstance(block_input, int) or isinstance(block_output, int):
        return (
            block_input if isinstance(block_input, int) else None,
            block_output if isinstance(block_output, int) else None,
        )

    combined = payload.get("BlockIO")
    if isinstance(combined, str) and "/" in combined:
        parts = [part.strip() for part in combined.split("/", maxsplit=1)]
        if len(parts) == 2:
            return parse_byte_value(parts[0]), parse_byte_value(parts[1])

    return None, None


# --------------------------------------------------------------------------
# Container-runtime queries
# --------------------------------------------------------------------------

def collect_container_stats(runtime: str, container_name: str) -> Optional[Dict[str, Any]]:
    """Collect one-shot stats for a running container."""
    result = subprocess.run(
        [runtime, "stats", "--no-stream", "--format", "{{json .}}", container_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None

    raw_line = result.stdout.strip().splitlines()[0]
    payload = json.loads(raw_line)
    rss_bytes, memory_limit_bytes = parse_memory_usage(payload)
    block_read_bytes, block_write_bytes = parse_block_io(payload)

    return {
        "cpu_percent": parse_cpu_percent(payload),
        # Cumulative CPU nanoseconds. The runtime's own CPU percent is an
        # average over the container's whole lifetime (CPU == AvgCPU), so it
        # smears any change in load; differencing this counter between two
        # samples is the only way to recover instantaneous use.
        "cpu_time_ns": parse_int_value(payload.get("CPUNano", "")),
        "rss_bytes": rss_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "num_pids": parse_int_value(payload.get("PIDs", "")),
        "block_read_bytes": block_read_bytes,
        "block_write_bytes": block_write_bytes,
        "status": "running",
    }


def inspect_container_status(runtime: str, container_name: str) -> str:
    """Inspect container status, tolerating removed containers."""
    result = subprocess.run(
        [runtime, "inspect", "--format", "{{.State.Status}}", container_name],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    if result.returncode != 0:
        stderr_text = (result.stderr or "").strip().lower()
        if "no such object" in stderr_text or "no such container" in stderr_text:
            return "removed"
        return "unavailable"
    return (result.stdout or "").strip() or "unknown"


# --------------------------------------------------------------------------
# Cgroup readers
# --------------------------------------------------------------------------

def resolve_cgroup_cpu_stat_path(runtime: str, container_name: str) -> Optional[Path]:
    """Resolve the container's cgroup v2 ``cpu.stat``.

    Reading the counter straight from the cgroup avoids a per-sample
    subprocess. That matters under contention: a ``podman stats`` fork can
    take seconds when the machine is saturated, and the counter it returns is
    then stale by that much, which shows up as a phantom dip in the
    instantaneous rate.

    Callers cache the result for the container's lifetime.
    """
    try:
        result = subprocess.run(
            [runtime, "inspect", container_name, "--format", "{{.State.CgroupPath}}"],
            capture_output=True, text=True, check=False, timeout=10,
        )
        rel = result.stdout.strip()
        if result.returncode == 0 and rel:
            root = Path("/sys/fs/cgroup")
            for candidate in (root / rel.lstrip("/") / "container" / "cpu.stat",
                              root / rel.lstrip("/") / "cpu.stat"):
                if os.access(candidate, os.R_OK):
                    return candidate
    except Exception:
        return None
    return None


def read_cgroup_cpu_time_ns(path: Path) -> Optional[int]:
    """Cumulative CPU nanoseconds from cgroup v2 ``cpu.stat``."""
    try:
        for line in path.read_text().splitlines():
            if line.startswith("usage_usec "):
                return int(line.split()[1]) * 1000
    except Exception:
        return None
    return None


def read_cgroup_throttling(path: Path) -> Dict[str, Any]:
    """Quota throttling counters from the same ``cpu.stat``.

    Populated only when a CPU quota is set, so these stay zero under load
    antagonists and carry the signal under ``controls.cpu.max_cores``: direct
    evidence the cap bit, and how much runnable time it removed, rather than
    inferring it from the shape of the CPU line.
    """
    out: Dict[str, Any] = {}
    try:
        for line in path.read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in ("nr_periods", "nr_throttled", "throttled_usec"):
                out[key] = int(value)
    except Exception:
        return {}
    return out


# --------------------------------------------------------------------------
# Stateful readers
# --------------------------------------------------------------------------

class GpuTelemetryReader:
    """Host-wide GPU telemetry via ``nvidia-smi``. Fail-soft, and latching.

    The first failure disables the reader for the rest of the run, so a host
    without ``nvidia-smi`` costs one failed subprocess rather than one per
    sample.

    These figures are whole-card: under GPU contention they include the load
    generator's own usage, so they cannot attribute consumption to the SLAM.
    """

    def __init__(self) -> None:
        self._disabled = False

    @property
    def disabled(self) -> bool:
        return self._disabled

    @staticmethod
    def empty() -> Dict[str, Any]:
        return {field: None for field in _GPU_FIELDS}

    def read(self) -> Dict[str, Any]:
        if self._disabled:
            return self.empty()

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu,memory.used,memory.total,"
                    "power.draw,temperature.gpu,clocks.current.sm",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
        except FileNotFoundError:
            self._disabled = True
            logger.warning("nvidia-smi not found; GPU telemetry will be recorded as None.")
            return self.empty()
        except Exception as exc:
            self._disabled = True
            logger.warning("nvidia-smi failed (%s); GPU telemetry disabled.", exc)
            return self.empty()

        if result.returncode != 0 or not (result.stdout or "").strip():
            return self.empty()

        first_line = result.stdout.strip().splitlines()[0]
        parts = [part.strip() for part in first_line.split(",")]
        if len(parts) < 7:
            return self.empty()

        def _float(text: str) -> Optional[float]:
            if not text or text.lower() in {"n/a", "[not supported]"}:
                return None
            try:
                return float(text)
            except ValueError:
                return None

        def _int(text: str) -> Optional[int]:
            value = _float(text)
            return int(value) if value is not None else None

        return {
            "gpu_util_percent": _float(parts[1]),
            "gpu_mem_used_mb": _int(parts[2]),
            "gpu_mem_total_mb": _int(parts[3]),
            "gpu_power_w": _float(parts[4]),
            "gpu_temp_c": _float(parts[5]),
            "gpu_sm_clock_mhz": _int(parts[6]),
        }


class ProcessGroupSampler:
    """Split a cgroup's CPU time and run-queue delay by workload.

    An in-container antagonist shares the SLAM's cgroup, so the cgroup-wide
    counter cannot say how much of a saturated machine the SLAM actually got.
    Walking the cgroup's own process list splits it. ``run_delay_ns`` (from
    schedstat) is the more useful half: time threads spent runnable but
    waiting for a CPU, which is starvation measured directly rather than
    inferred from a drop rate.

    Reads the host's /proc -- ``cgroup.procs`` lists host PIDs, so this needs
    no exec into the container.

    Stateful because deltas are taken against the previous pass. Rate-limited
    to ``period_s`` because the walk is ~1000 small reads under a heavy
    antagonist: the counters are cumulative, so a coarser cadence costs
    resolution, not accuracy.
    """

    def __init__(self, period_s: float = PROCESS_WALK_PERIOD_S) -> None:
        self._period_s = period_s
        # Per-PID cumulative CPU/run-delay from the previous walk, so deltas
        # survive processes appearing and exiting between passes.
        self._snapshot: Dict[int, tuple] = {}
        self._last_walk: Optional[float] = None
        self._last_walk_at: Optional[float] = None

    def sample_if_due(self, cpu_stat_path: Path) -> Optional[Dict[str, Any]]:
        """Walk the cgroup if the rate limit allows, else return None."""
        now = time.monotonic()
        if self._last_walk is not None and (now - self._last_walk) < self._period_s:
            return None
        groups = self.sample(cpu_stat_path)
        self._last_walk = time.monotonic()
        return groups

    def sample(self, cpu_stat_path: Path) -> Optional[Dict[str, Any]]:
        procs_file = cpu_stat_path.parent / "cgroup.procs"
        try:
            pids = [int(p) for p in procs_file.read_text().split()]
        except Exception:
            return None

        # Group totals are NOT differenced directly: processes come and go
        # (ORB-SLAM3 churns threads, hard during a map reset), and an exited
        # process removes its accumulated time from the sum, which reads as
        # negative CPU use. Deltas are taken per PID against the previous pass
        # and only then summed.
        current: Dict[int, tuple] = {}
        groups: Dict[str, Any] = {
            "slam": {"cpu_time_ns": 0, "run_delay_ns": 0, "procs": 0, "threads": 0},
            "load": {"cpu_time_ns": 0, "run_delay_ns": 0, "procs": 0, "threads": 0},
        }
        ticks = os.sysconf("SC_CLK_TCK") or 100
        for pid in pids:
            try:
                stat = (Path("/proc") / str(pid) / "stat").read_text()
            except Exception:
                continue  # exited between listing and reading
            # comm is parenthesised and may contain spaces; split on the last ')'
            close = stat.rfind(")")
            open_paren = stat.find("(")
            if close < 0 or open_paren < 0:
                continue
            comm = stat[open_paren + 1:close]
            fields = stat[close + 2:].split()
            if len(fields) < 13:
                continue
            # utime/stime are fields 14/15 in proc(5), i.e. index 11/12 after state
            cpu_ticks = int(fields[11]) + int(fields[12])
            cpu_ns = int(cpu_ticks / ticks * 1e9)
            key = "load" if comm.startswith("stress-ng") else "slam"
            g = groups[key]
            g["procs"] += 1
            # schedstat is per-thread, so sum the thread group.
            delay_ns = 0
            try:
                tids = list((Path("/proc") / str(pid) / "task").iterdir())
            except Exception:
                tids = []
            for tid in tids:
                try:
                    parts = (tid / "schedstat").read_text().split()
                except Exception:
                    continue
                if len(parts) >= 2:
                    delay_ns += int(parts[1])
                    g["threads"] += 1

            current[pid] = (key, cpu_ns, delay_ns)
            prev = self._snapshot.get(pid)
            if prev is not None and prev[0] == key:
                # Counters only advance; a decrease means PID reuse, so skip it
                # rather than contribute a negative interval.
                d_cpu, d_delay = cpu_ns - prev[1], delay_ns - prev[2]
                if d_cpu >= 0:
                    g["cpu_time_ns"] += d_cpu
                if d_delay >= 0:
                    g["run_delay_ns"] += d_delay
            # A PID seen for the first time contributes nothing this pass: its
            # counter started at an unknown point before we looked. It counts
            # fully from the next pass onward.

        self._snapshot = current
        # These are per-interval deltas, not cumulative totals.
        groups["interval_s"] = (
            None if self._last_walk_at is None
            else time.monotonic() - self._last_walk_at
        )
        self._last_walk_at = time.monotonic()
        return groups


class GpuProcessSampler:
    """Per-process GPU attribution via NVML. Fail-soft and latching.

    The whole-card figures from ``nvidia-smi`` cannot say how much of the GPU
    the SLAM itself got, only that the card was busy. Under GPU contention that
    is the difference between a measured dose and a calibrated one, and it is
    also the only way to record the SLAM's own peak VRAM as a clean-run
    reference (preflight P4/P5).

    Two NVML calls per sample:

    - ``nvmlDeviceGetComputeRunningProcesses`` gives per-PID VRAM held now.
    - ``nvmlDeviceGetProcessUtilization`` gives per-PID SM and memory-controller
      utilisation over the driver's recent sample window. It raises NotFound
      when the sample buffer is empty (an idle card), which is normal and not
      an error.

    PIDs are host PIDs, the same namespace as ``cgroup.procs``, so the caller
    can split SLAM from competitor with the process list it already walks.
    """

    def __init__(self) -> None:
        self._disabled = False
        self._nvml = None
        self._handle = None
        self._last_ts = 0

    @property
    def disabled(self) -> bool:
        return self._disabled

    def _init(self) -> bool:
        if self._handle is not None:
            return True
        try:
            import pynvml  # noqa: PLC0415 — optional, and only on a GPU host
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            return True
        except Exception as exc:
            self._disabled = True
            logger.warning("NVML unavailable (%s); per-process GPU telemetry disabled.", exc)
            return False

    def read(self) -> Optional[Dict[int, Dict[str, Any]]]:
        """{host_pid: {vram_mb, sm_util, mem_util}} or None if unavailable."""
        if self._disabled or not self._init():
            return None
        N = self._nvml
        out: Dict[int, Dict[str, Any]] = {}
        try:
            try:
                procs = N.nvmlDeviceGetComputeRunningProcesses_v3(self._handle)
            except AttributeError:
                procs = N.nvmlDeviceGetComputeRunningProcesses(self._handle)
            for pr in procs:
                out[int(pr.pid)] = {
                    # MiB, not decimal MB, despite the field name. The
                    # peak-RSS reference in gen_e2_configs is decimal MB,
                    # so the two "mb" fields are in DIFFERENT units.
                    "vram_mb": int((pr.usedGpuMemory or 0) / 1048576),
                    "sm_util": None,
                    "mem_util": None,
                }
        except Exception as exc:
            self._disabled = True
            logger.warning("NVML process query failed (%s); disabling.", exc)
            return None

        # Utilisation is best-effort on top: NotFound simply means the driver's
        # sample buffer held nothing for this window, which is not a failure.
        try:
            for sample in N.nvmlDeviceGetProcessUtilization(self._handle, self._last_ts):
                pid = int(sample.pid)
                entry = out.setdefault(pid, {"vram_mb": None, "sm_util": None, "mem_util": None})
                entry["sm_util"] = int(sample.smUtil)
                entry["mem_util"] = int(sample.memUtil)
                self._last_ts = max(self._last_ts, int(sample.timeStamp))
        except Exception:
            pass
        return out or None


def read_cgroup_memory_events(cpu_stat_path: Path) -> Dict[str, Any]:
    """OOM accounting from cgroup v2 ``memory.events``, beside ``cpu.stat``.

    ``oom`` counts times the group hit its limit and had to reclaim or kill;
    ``oom_kill`` counts processes the kernel actually killed. This is the only
    positive evidence that a HOST memory cap is what ended a run: exit 139 and
    exit 137 both appear on healthy and destroyed runs alike, so an exit code
    cannot attribute a death to the axis under test (preflight P6, which E2
    depends on because its whole product is "which axis killed it").
    """
    out: Dict[str, Any] = {}
    try:
        for line in (cpu_stat_path.parent / "memory.events").read_text().splitlines():
            key, _, value = line.partition(" ")
            if key in ("oom", "oom_kill", "oom_group_kill", "max", "high"):
                out[key] = int(value)
    except Exception:
        return {}
    return out
