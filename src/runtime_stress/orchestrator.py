"""Runtime-stress orchestration for one SLAM execution."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

import logging

from .controllers import CpuQuotaController, DockerCpuController, DockerMemoryController, ResourceController
from .load_controller import PodmanLoadController
from .podman_controllers import PodmanCpuController, PodmanIoController, PodmanMemoryController
from .hami_controller import (
    GpuHamiController,
    HAMI_LIB_HOST_PATH,
    hami_launch_devices,
    hami_launch_env,
    hami_launch_mounts,
)
from .models import RuntimeStressRequest
from .telemetry import (
    GpuProcessSampler,
    GpuTelemetryReader,
    ProcessGroupSampler,
    collect_container_stats,
    inspect_container_status,
    read_cgroup_cpu_time_ns,
    read_cgroup_memory_events,
    read_cgroup_throttling,
    resolve_cgroup_cpu_stat_path,
)

logger = logging.getLogger(__name__)

CONTAINER_TARGET_KINDS = {"docker_container", "podman_container"}

# How long a frame-anchored run may go without the SLAM publishing its
# progress file before the run is marked invalid (controller_error). Sized
# to absorb the slowest model-load observed (MASt3R checkpoint load), which
# precedes the first progress write.
FRAME_PROGRESS_GRACE_S = 300.0

class _HarnessLogCapture(logging.Handler):
    """Persist the harness's own output next to the SLAM's, and promote every
    warning into a run event.

    Hard failures were already durable: they raise, `_run_loop` catches them, and
    `_record_event("controller_error", ...)` puts them in stress_events.json. Soft
    degradations were not. Lines like "initialized_flag stayed 0 ... skipping
    mutation for this phase" or "expect read_iops/write_iops caps to silently
    no-op" went to a console logger with no file handler anywhere, so the harness
    kept a detailed record of what the SLAM did and none of what it did to the
    SLAM.

    That is the dangerous half. A run that fails loudly leaves evidence. A run
    where the harness quietly did something other than what the config asked for
    left nothing, and "no warnings found" was indistinguishable from "warnings
    were never recorded anywhere".

    WARNING and above is also appended to the run's events, so a degradation is
    machine-readable rather than something a human has to notice scrolling past.
    verify_treatment can then refuse a run that carries one.
    """

    def __init__(self, path: Path, events: List[Dict[str, Any]], elapsed_fn):
        super().__init__(level=logging.INFO)
        self._events = events
        self._elapsed_fn = elapsed_fn
        self._stream = open(path, "a", encoding="utf-8")
        self.setFormatter(
            logging.Formatter(
                "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    def emit(self, record: logging.LogRecord) -> None:
        # Never raise out of a log handler: a broken audit trail must not be able
        # to kill a run that is otherwise fine.
        try:
            self._stream.write(self.format(record) + "\n")
            self._stream.flush()
        except Exception:
            pass
        if record.levelno < logging.WARNING:
            return
        try:
            self._events.append(
                {
                    "kind": "harness_degraded",
                    "elapsed_s": round(self._elapsed_fn(), 6),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
            )
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._stream.close()
        except Exception:
            pass
        super().close()


def _container_runtime_binary(target_kind: str) -> str:
    """Return the container runtime binary name for a container target_kind."""
    if target_kind == "podman_container":
        return "podman"
    return "docker"


@dataclass
class RuntimeStressOrchestrator:
    """Apply runtime-stress phases to one running SLAM process."""

    request: RuntimeStressRequest
    output_dir: Path
    trace: List[Dict[str, Any]] = field(default_factory=list)
    events: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        self._controllers: List[ResourceController] = []
        self._has_cpu_controls = any(
            phase.controls.cpu is not None and phase.controls.cpu.max_cores is not None
            for phase in self.request.phases
        )
        self._has_memory_controls = any(
            phase.controls.memory is not None and phase.controls.memory.max_mb is not None
            for phase in self.request.phases
        )
        # Check both the launch-time-seeded request.gpu (env-var injection)
        # AND any per-phase gpu controls. With runtime mutation supported,
        # a scenario may declare GPU caps only in non-first phases; the
        # controller still needs to be built so apply() can mutate at the
        # phase boundary.
        self._has_gpu_controls = (
            self.request.gpu is not None
            and (
                self.request.gpu.vram_limit_mb is not None
                or self.request.gpu.sm_limit_percent is not None
            )
        ) or any(
            phase.controls.gpu is not None
            and (
                phase.controls.gpu.vram_limit_mb is not None
                or phase.controls.gpu.sm_limit_percent is not None
            )
            for phase in self.request.phases
        )
        self._has_io_controls = any(
            phase.controls.io is not None
            and (
                phase.controls.io.read_bps is not None
                or phase.controls.io.write_bps is not None
                or phase.controls.io.read_iops is not None
                or phase.controls.io.write_iops is not None
            )
            for phase in self.request.phases
        )
        self._has_load_controls = any(
            phase.controls.load is not None for phase in self.request.phases
        )
        self._gpu_telemetry = GpuTelemetryReader()
        # Per-process GPU attribution. Whole-card figures cannot say what the
        # SLAM itself got; this is the GPU counterpart of the process walk.
        self._gpu_processes = GpuProcessSampler()
        self._io_controller: Optional[PodmanIoController] = None
        self._load_controller: Optional[PodmanLoadController] = None

        self._process = None
        self._process_info = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._current_phase_index: Optional[int] = None
        self._started_at: Optional[float] = None
        self._controller_error: Optional[str] = None
        self._finalized = False
        self._log_capture: Optional[_HarnessLogCapture] = None
        self._target_kind: Optional[str] = None
        self._target_metadata: Dict[str, Any] = {}

        # Frame-anchored phases: advance by the SLAM's live frame count
        # (published by the deadline harness to SAL_PROGRESS_PATH) instead
        # of wall-clock elapsed time, so a control change lands at the same
        # trajectory point regardless of the paced frame rate.
        self._frame_anchored = self.request.frame_anchored
        self._progress_path: Optional[str] = None
        self._last_frame = 0
        # Fail-loud guard for frame-anchored runs: if the SLAM never
        # publishes the progress file (entry point not wired, or a stale
        # baked-in entry script), phases would silently stay at index 0 for
        # the whole run and the results would look like a fully-stressed
        # run that never happened. Track whether progress was ever seen so
        # the run can be marked invalid instead.
        self._progress_seen = False
        self._progress_stall_reported = False

        # Splits the cgroup's CPU time between the SLAM and any in-container
        # load. Stateful (per-PID deltas) and self-rate-limiting.
        self._process_groups = ProcessGroupSampler()

    def attach_process(
        self,
        process,
        target_kind: str,
        target_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Attach the orchestrator to a spawned process."""
        if target_kind not in {"host_process_group"} | CONTAINER_TARGET_KINDS:
            raise RuntimeError(
                f"Runtime stress does not support execution target kind '{target_kind}'."
            )

        self._process = process
        self._target_kind = target_kind
        self._target_metadata = dict(target_metadata or {})

        # Installed before the controllers are built, so warnings raised while
        # they PREPARE are captured too. That is where several of the "this cap
        # may not do anything" messages are emitted.
        self._install_log_capture()

        if target_kind == "host_process_group":
            self._process_info = psutil.Process(process.pid)
            self._process_info.cpu_percent(None)
            psutil.cpu_percent(None)
        else:
            self._process_info = None

        self._controllers = self._build_controllers(target_kind)

        for controller in self._controllers:
            try:
                controller.prepare(
                    process,
                    target_kind=target_kind,
                    target_metadata=self._target_metadata,
                )
            except Exception as exc:
                # Same fail-loud convention as apply() failures in _run_loop:
                # attribute the failure in the summary, kill the already-
                # spawned SLAM so it cannot finish as a valid-looking
                # unstressed run, then propagate.
                self._controller_error = f"controller prepare failed: {exc}"
                self._record_event(
                    "controller_error", elapsed_s=0.0, message=str(exc)
                )
                try:
                    self._terminate_target()
                except Exception:
                    pass
                raise

        self._started_at = time.monotonic()
        if self._frame_anchored:
            # Set by the pipeline's _realtime_env before the SLAM launched.
            self._progress_path = os.environ.get("SAL_PROGRESS_PATH")
        attach_message = "runtime stress attached"
        if target_kind in CONTAINER_TARGET_KINDS:
            runtime_label = _container_runtime_binary(target_kind)
            container_name = self._target_metadata.get("container_name")
            if container_name:
                attach_message = f"runtime stress attached to {runtime_label} container '{container_name}'"
        self._record_event("attach", elapsed_s=0.0, message=attach_message)
        self._thread = threading.Thread(target=self._run_loop, name="runtime-stress", daemon=True)
        self._thread.start()

    def finalize(
        self,
        execution_ok: bool,
        trajectory_found: bool,
    ) -> None:
        """Stop orchestration and write artifacts."""
        if self._finalized:
            return

        self._finalized = True
        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5)

        # Snapshot any controller-side artifacts that cleanup() will clear,
        # so finalize can include them in the summary below.
        io_target_devices_snapshot: Optional[List[Dict[str, Any]]] = None
        if self._io_controller is not None:
            io_target_devices_snapshot = self._io_controller.target_device_summaries()
        load_antagonists_snapshot: Optional[List[Dict[str, Any]]] = None
        if self._load_controller is not None:
            load_antagonists_snapshot = self._load_controller.antagonist_summaries()

        for controller in self._controllers:
            try:
                controller.release()
            except Exception:
                pass
            try:
                controller.cleanup()
            except Exception:
                pass

        # After release/cleanup, so anything they warn about is still captured,
        # and before the events file is written, so the list is final.
        self._remove_log_capture()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trace_path = self.output_dir / "stress_trace.json"
        events_path = self.output_dir / "stress_events.json"
        summary_path = self.output_dir / "stress_summary.json"

        trace_path.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
        events_path.write_text(json.dumps(self.events, indent=2), encoding="utf-8")

        exit_code = None
        if self._process is not None:
            exit_code = self._process.returncode

        duration_s = round(self._elapsed_s(), 3) if self._started_at is not None else None

        summary = {
            "scenario_name": self.request.scenario_name,
            "target_kind": self._target_kind,
            "target_metadata": self._target_metadata,
            "execution_ok": execution_ok,
            "trajectory_found": trajectory_found,
            "exit_code": exit_code,
            "duration_s": duration_s,
            "pid": self._process.pid if self._process is not None else None,
            "controller_error": self._controller_error,
            "phase_count": len(self.request.phases),
            "samples": len(self.trace),
            "events": len(self.events),
        }
        if io_target_devices_snapshot is not None:
            summary["io_target_devices"] = io_target_devices_snapshot
        if load_antagonists_snapshot is not None:
            summary["load_antagonists"] = load_antagonists_snapshot
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def _run_loop(self) -> None:
        sample_period_s = max(self.request.telemetry_sample_period_ms / 1000.0, 0.05)
        try:
            while not self._stop_event.is_set():
                elapsed = self._elapsed_s()
                current_frame = None
                # Captured BEFORE the read, which advances _last_frame in
                # place. The skip guard reports the jump, so it needs both ends.
                prev_frame = self._last_frame
                if self._frame_anchored:
                    current_frame = self._read_progress_frame()
                    phase_index = self._phase_index_for_frame(current_frame)
                    self._check_progress_stall(elapsed)
                else:
                    phase_index = self._phase_index_for_elapsed(elapsed)
                if phase_index != self._current_phase_index:
                    self._check_phase_skip(
                        phase_index, prev_frame, current_frame, elapsed
                    )
                    self._apply_phase(phase_index, elapsed, frame=current_frame)

                self._sample(elapsed)

                if self._process is not None and self._process.poll() is not None:
                    break

                # Flush the trace to disk periodically, not only in finalize().
                # A run that STALLS under a cap is exactly the run most likely to be
                # killed at a timeout, and it is also the one whose telemetry decides
                # whether it was stall-bound or merely slow. Writing only at the end
                # meant the most severe rung -- the one most likely to bind -- was
                # the one guaranteed to have no evidence.
                if len(self.trace) % 20 == 0:
                    self._flush_trace()
                self._stop_event.wait(sample_period_s)
        except Exception as exc:
            self._controller_error = str(exc)
            self._record_event(
                "controller_error",
                elapsed_s=self._elapsed_s(),
                message=str(exc),
            )
            self._terminate_target()
        finally:
            self._sample(self._elapsed_s())

    def _build_controllers(self, target_kind: str) -> List[ResourceController]:
        """Build resource controllers for the attached execution target."""
        controllers: List[ResourceController] = []

        if self._has_cpu_controls:
            if target_kind == "host_process_group":
                controllers.append(CpuQuotaController())
            elif target_kind == "docker_container":
                controllers.append(DockerCpuController())
            elif target_kind == "podman_container":
                controllers.append(PodmanCpuController())
            else:
                raise RuntimeError(f"Unsupported runtime-stress target kind '{target_kind}'.")

        if self._has_memory_controls:
            if target_kind == "docker_container":
                controllers.append(DockerMemoryController())
            elif target_kind == "podman_container":
                controllers.append(
                    PodmanMemoryController(
                        preapplied_max_mb=self.constant_memory_cap_mb()
                    )
                )
            else:
                raise RuntimeError(
                    f"Memory runtime stress is only supported for container targets, got '{target_kind}'."
                )

        if self._has_gpu_controls:
            if target_kind in CONTAINER_TARGET_KINDS:
                controllers.append(GpuHamiController())
            else:
                raise RuntimeError(
                    f"GPU runtime stress (HAMi) does not support execution target kind "
                    f"'{target_kind}' in v1. Host-process GPU isolation is deferred to the "
                    "PhotoSLAM Podman migration cycle."
                )

        if self._has_io_controls:
            if target_kind == "podman_container":
                io_controller = PodmanIoController()
                self._io_controller = io_controller
                controllers.append(io_controller)
            else:
                raise RuntimeError(
                    f"IO runtime stress is only supported for podman_container in v1, "
                    f"got '{target_kind}'. Docker support is blocked by missing per-device "
                    "IO mutation in 'docker update'; host-process IO is deferred to a "
                    "follow-up cycle."
                )

        if self._has_load_controls:
            if target_kind == "podman_container":
                # Reuse the controller constructed by prewarm_load_antagonists()
                # (the hold point) so the already-ready GPU antagonist carries
                # over; construct fresh only if no prewarm happened. Appended
                # LAST so any residual slow prepare work follows the fast cap
                # controllers' attach loops.
                load_controller = self._load_controller or PodmanLoadController(
                    load_specs=self._compiled_load_specs()
                )
                self._load_controller = load_controller
                controllers.append(load_controller)
            else:
                raise RuntimeError(
                    "Load-antagonist runtime stress is only supported for "
                    f"podman_container in v1, got '{target_kind}'."
                )

        return controllers

    def _compiled_load_specs(self):
        return [
            phase.controls.load
            for phase in self.request.phases
            if phase.controls.load is not None
        ]

    def prewarm_load_antagonists(self) -> None:
        """Hold point: ready the GPU load antagonist BEFORE the SLAM launches.

        Called by the SLAM base right after the session is constructed and
        before the SLAM container spawns. Without this, the antagonist's
        CUDA init (5-15 s) overlaps the SLAM's startup and fast-starting
        SLAMs get an uncontended head in the stress phase. No-op unless the
        scenario declares a GPU load antagonist. Fail loud: a prewarm
        failure aborts the run before any SLAM exists, with the error
        attributed via controller_error for the finalize summary.
        """
        if not self._has_load_controls:
            return
        if self._load_controller is None:
            self._load_controller = PodmanLoadController(
                load_specs=self._compiled_load_specs()
            )
        try:
            started = time.monotonic()
            # SAM 3 needs ~9 s to load. Readying it here rather than at the
            # phase boundary is what keeps the run fully contended; without it
            # the SLAM runs through that load uncontended.
            if self._load_controller.prewarm_segmentation():
                self._record_event(
                    "prewarm",
                    elapsed_s=0.0,
                    message=(
                        "segmentation co-tenant ready before SLAM launch "
                        f"({time.monotonic() - started:.1f}s)"
                    ),
                )
            started = time.monotonic()
            if self._load_controller.prewarm_gpu():
                self._record_event(
                    "prewarm",
                    elapsed_s=0.0,
                    message=(
                        "GPU load antagonist ready before SLAM launch "
                        f"({time.monotonic() - started:.1f}s)"
                    ),
                )
        except Exception as exc:
            self._controller_error = f"controller prewarm failed: {exc}"
            self._record_event("controller_error", elapsed_s=0.0, message=str(exc))
            try:
                self._load_controller.cleanup()
            except Exception:
                pass
            raise

    def gpu_launch_config(self) -> Dict[str, Any]:
        """Return env/mounts the SLAM wrapper must inject for HAMi to take effect.

        When no GPU controls are declared, returns empty env/mounts lists so
        callers can merge unconditionally. When GPU controls are present,
        returns the HAMi env vars and the libvgpu.so bind mount, plus the
        host path of the HAMi library (for diagnostics).
        """
        if not self._has_gpu_controls or self.request.gpu is None:
            return {"env": {}, "mounts": [], "devices": []}

        mounts = hami_launch_mounts()
        return {
            "env": hami_launch_env(self.request.gpu),
            "mounts": mounts,
            "devices": hami_launch_devices(),
            "lib_path": mounts[0][0] if mounts else HAMI_LIB_HOST_PATH,
        }

    def memory_launch_config(self) -> Dict[str, Any]:
        """Apply a CONSTANT memory cap at container launch instead of mid-run.

        ``podman update --memory`` hung on one system of eight when lowering a
        limit, leaving the cap undelivered while the run looked normal. The
        condition is not fully characterised -- orbslam3i's cap landed at 136 MB
        against a 544 MB reference, so "cannot lower below current usage" is NOT
        the rule -- but a mechanism that silently fails on one allocation pattern
        will eventually fail on an untried one.

        Only used when every phase declaring a memory cap declares the SAME one,
        which is true of a cap ladder (constant per run) and false of a
        time-varying scenario, where the mid-run pathway is still required.
        """
        cap = self.constant_memory_cap_mb()
        if cap is None:
            return {"env": {}, "mounts": [], "devices": [], "run_flags": []}
        return {"env": {}, "mounts": [], "devices": [],
                "run_flags": ["--memory", f"{cap}m"]}

    def constant_memory_cap_mb(self) -> Optional[int]:
        """The one memory cap every capped phase shares, or None.

        Single source of truth for both the launch flag and the controller that
        must then skip re-applying it. Reading it twice from separate logic is
        how the flag and the skip drift apart, which would put the run back on
        the update path it was moved off.
        """
        caps = {
            int(phase.controls.memory.max_mb)
            for phase in self.request.phases
            if phase.controls.memory is not None
            and phase.controls.memory.max_mb is not None
        }
        return caps.pop() if len(caps) == 1 else None

    def load_launch_config(self) -> Dict[str, Any]:
        """Return mounts the SLAM wrapper must inject for in-container load.

        The in-container load mode execs the framework's static stress-ng
        into the SLAM's own cgroup; the binary reaches the container via
        this launch-time bind mount (same pattern as HAMi's libvgpu.so).
        Empty when no phase declares in_container load, so callers can
        merge unconditionally. Binary presence is enforced fail-loud by the
        load controller's preflight, not here (the base swallows launch
        config errors into empty extras).
        """
        from .load_controller import (
            STATIC_STRESS_NG_HOST_PATH,
            STRESS_NG_CONTAINER_PATH,
        )

        needs_in_container = self._has_load_controls and any(
            phase.controls.load is not None
            and phase.controls.load.in_container is not None
            for phase in self.request.phases
        )
        if not needs_in_container:
            return {"env": {}, "mounts": [], "devices": []}
        return {
            "env": {},
            "mounts": [(str(STATIC_STRESS_NG_HOST_PATH), STRESS_NG_CONTAINER_PATH, "ro")],
            "devices": [],
        }

    def _elapsed_s(self) -> float:
        if self._started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self._started_at)

    def _phase_index_for_elapsed(self, elapsed_s: float) -> int:
        cumulative = 0.0
        last_index = len(self.request.phases) - 1
        for index, phase in enumerate(self.request.phases):
            cumulative += phase.duration_s
            if elapsed_s < cumulative:
                return index
        return last_index

    def _phase_index_for_frame(self, current_frame: int) -> int:
        """Pick the active phase by the SLAM's current sampled-frame index.

        Each frame-anchored phase ends at its ``until_frame`` boundary; the
        SLAM is in phase ``i`` while ``current_frame < until_frame[i]``. Past
        the last boundary it clamps to the final phase, mirroring the
        wall-clock path.

        **BOUNDARIES ARE IN SAMPLED FRAMES, NOT STREAM FRAMES.** The counter is
        what the deadline harness publishes, which for a strided SLAM is far
        below the dataset's frame budget: DPVO and DROID sample every 2nd frame
        on TUM, so a 500-frame stream produces a counter that tops out near 250.
        A boundary written in stream frames is then unreachable, the run never
        leaves that phase, and it LOOKS like a completed multi-phase run.

        Measured, first pass: dpvslam's release boundary at stream-frame
        250 was never crossed (counter reached 247), so the cap stayed on for
        the whole second half and throttling read 98-100% in what should have
        been the released phase. orbslam3i, stride 1, reached 499 and behaved
        correctly. The failure is silent because the stall guard only catches a
        counter that never MOVES, not one that never ARRIVES.
        """
        last_index = len(self.request.phases) - 1
        for index, phase in enumerate(self.request.phases):
            if current_frame < phase.until_frame:
                return index
        return last_index

    def _check_phase_skip(
        self,
        phase_index: int,
        prev_frame: int,
        current_frame: Optional[int],
        elapsed_s: float,
    ) -> None:
        """Fail loud when a phase's controls were NEVER applied.

        Phase boundaries are evaluated only when a telemetry sample is taken.
        If the SLAM's frame counter advances further between two samples than a
        phase is wide, the orchestrator moves straight past that phase and its
        controls never actuate -- yet the run completes, writes a trajectory,
        and reads as a successful multi-phase experiment.

        This is NOT the stall guard's case. There the counter never moves; here
        it moves too far. Both leave phase controls unapplied, and neither is
        visible in the trajectory.

        Measured, second pass, at a 500 ms sample period:

        - ``gigaslam`` published frame 16 for 16 consecutive samples (~9.5 s),
          then 116. Its squeeze window was 60-100, so the phase was stepped
          over entirely: the trace holds ``clean_before`` and ``released`` and
          no ``squeeze`` at all. The cap never applied.
        - ``dpvslam`` advanced 62 -> 131 across one sample, leaving a 75-125
          window with a single sample in it and 43% throttling where a real
          0.5-core squeeze on that system throttles ~98%.
        - ``orbslam3i``, whose counter never repeats more than twice, crossed
          every boundary correctly. Its result stands.

        A phase is skipped when the index advances by more than one. Frame
        anchoring is what makes this reachable -- wall-clock phases advance
        with the sampler itself -- so the guard is scoped to it.
        """
        if not self._frame_anchored or self._current_phase_index is None:
            return
        if phase_index <= self._current_phase_index + 1:
            return
        skipped = [
            self.request.phases[i].name
            for i in range(self._current_phase_index + 1, phase_index)
        ]
        message = (
            f"phase(s) {skipped} were skipped entirely: the frame counter "
            f"jumped from {prev_frame} past their boundaries to "
            f"{current_frame} between two telemetry samples, so their "
            "controls NEVER applied. Results from this run are invalid for "
            "any claim about those phases. Widen the phase, shorten "
            "telemetry.sample_period_ms, or use wall-clock phases for a SLAM "
            "that publishes frame progress in bursts."
        )
        # Never overwrite an earlier error; the first failure is the cause.
        if not self._controller_error:
            self._controller_error = message
        self._record_event("error", elapsed_s=elapsed_s, message=message)
        logger.error(message)

    def _check_progress_stall(self, elapsed_s: float) -> None:
        """Fail loud when a frame-anchored run never publishes progress.

        Without the progress file, frame-anchored phases stay at index 0 for
        the whole run and the results would look like a fully-stressed run
        that never happened. Mark the run invalid (controller_error) once,
        after a grace window sized to the slowest model load.
        """
        if not self._frame_anchored:
            return
        if (
            self._progress_seen
            or self._progress_stall_reported
            or elapsed_s <= FRAME_PROGRESS_GRACE_S
        ):
            return
        self._progress_stall_reported = True
        self._controller_error = (
            "frame-anchored phases configured but no progress file "
            f"appeared within {FRAME_PROGRESS_GRACE_S:.0f}s "
            f"(SAL_PROGRESS_PATH={self._progress_path!r}); the SLAM "
            "entry point is not publishing frame progress, so later "
            "phases will never apply. Results from this run are "
            "invalid: only phase 0 controls were active."
        )
        self._record_event("error", elapsed_s=elapsed_s, message=self._controller_error)
        logger.error(self._controller_error)

    def _flush_trace(self) -> None:
        """Write the trace so far, atomically. Best-effort: never kill a run."""
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.output_dir / "stress_trace.json.tmp"
            tmp.write_text(json.dumps(self.trace, indent=2), encoding="utf-8")
            os.replace(tmp, self.output_dir / "stress_trace.json")
        except Exception:
            pass

    def _sample_progress_frame(self) -> Optional[int]:
        """Live frame index for telemetry, independent of phase anchoring.

        ``_progress_path`` is only populated from SAL_PROGRESS_PATH, which is
        set for FRAME-ANCHORED scenarios. Every deadline run writes the same
        file into its own output directory regardless, so gating the telemetry
        sample on that env var recorded nothing for wall-clock scenarios.
        Fall back to the run's own file.
        """
        path = self._progress_path or str(Path(self.output_dir) / "deadline_progress.json")
        try:
            with open(path) as f:
                return int(json.load(f).get("frame"))
        except (OSError, ValueError, TypeError):
            return None

    def _read_progress_frame(self) -> int:
        """Read the SLAM's live frame index from the progress file.

        Returns the last successfully-read frame on a missing, partial, or
        unparseable file, so a transient read never rewinds phase selection
        or crashes the monitor loop. Frame counts only advance.
        """
        if not self._progress_path:
            return self._last_frame
        try:
            with open(self._progress_path) as f:
                frame = int(json.load(f).get("frame", self._last_frame))
        except (OSError, ValueError, TypeError):
            return self._last_frame
        self._progress_seen = True
        if frame > self._last_frame:
            self._last_frame = frame
        return self._last_frame

    def _apply_phase(
        self, phase_index: int, elapsed_s: float, frame: Optional[int] = None
    ) -> None:
        phase = self.request.phases[phase_index]
        for controller in self._controllers:
            controller.apply(phase.controls)

        self._current_phase_index = phase_index
        self._record_event(
            "phase_enter",
            elapsed_s=elapsed_s,
            message=phase.name,
            frame=frame,
        )

    def _cpu_stat_path(self, runtime: str, container_name: str) -> Optional[Path]:
        """Resolve the container's cgroup ``cpu.stat``, caching only success.

        A failure is NOT cached. The first telemetry tick can fire before the
        container's cgroup is visible to ``inspect``, and caching that miss
        silently disabled the read-instant, the throttling counters, the
        per-workload walk and the GPU split for the WHOLE run, while
        ``container_cpu_time_ns`` kept being populated from the podman-stats
        fallback so the trace still looked healthy. Preflight P1 hit this on
        8 runs out of 8.

        Retrying costs one ``inspect`` per tick until it resolves, then nothing.
        """
        cached = getattr(self, "_cpu_stat_path_cache", None)
        if cached is not None:
            return cached
        resolved = resolve_cgroup_cpu_stat_path(runtime, container_name)
        if resolved is not None:
            self._cpu_stat_path_cache = resolved
        return resolved

    def _sample(self, elapsed_s: float) -> None:
        if self._target_kind in CONTAINER_TARGET_KINDS:
            self._sample_container(elapsed_s)
            return

        if self._process_info is None:
            return

        process_alive = self._process is not None and self._process.poll() is None
        phase_name = None
        if self._current_phase_index is not None:
            phase_name = self.request.phases[self._current_phase_index].name

        try:
            cpu_percent = self._process_info.cpu_percent(None)
            memory_info = self._process_info.memory_info()
            num_threads = self._process_info.num_threads()
            status = self._process_info.status()
        except Exception:
            cpu_percent = None
            memory_info = None
            num_threads = None
            status = "unavailable"

        self.trace.append(
            {
                "elapsed_s": round(elapsed_s, 6),
                "phase": phase_name,
                "target_kind": self._target_kind,
                "process_alive": process_alive,
                "process_cpu_percent": cpu_percent,
                "system_cpu_percent": psutil.cpu_percent(None),
                "rss_bytes": None if memory_info is None else memory_info.rss,
                "num_threads": num_threads,
                "status": status,
            }
        )

    def _sample_container(self, elapsed_s: float) -> None:
        """Sample runtime state for a Docker- or Podman-backed execution target."""
        runtime = _container_runtime_binary(self._target_kind or "docker_container")
        container_name = self._target_metadata.get("container_name")
        process_alive = self._process is not None and self._process.poll() is None
        phase_name = None
        if self._current_phase_index is not None:
            phase_name = self.request.phases[self._current_phase_index].name

        cpu_percent = None
        cpu_time_ns = None
        rss_bytes = None
        memory_limit_bytes = None
        num_pids = None
        block_read_bytes = None
        block_write_bytes = None
        status = "unavailable"

        # Read the CPU counter straight from the cgroup FIRST, and stamp the
        # exact moment it was read. Everything below may take seconds under
        # contention; pairing a counter with a timestamp taken after that
        # latency is what produces phantom dips in the computed rate.
        cpu_time_at_s = None
        throttling: Dict[str, Any] = {}
        memory_events: Dict[str, Any] = {}
        process_groups = None
        if isinstance(container_name, str) and container_name.strip():
            stat_path = self._cpu_stat_path(runtime, container_name.strip())
            if stat_path is not None:
                measured = read_cgroup_cpu_time_ns(stat_path)
                if measured is not None:
                    cpu_time_ns = measured
                    cpu_time_at_s = (
                        time.monotonic() - self._started_at
                        if self._started_at is not None else elapsed_s
                    )
                throttling = read_cgroup_throttling(stat_path)
                memory_events = read_cgroup_memory_events(stat_path)
                # The sampler rate-limits itself; the orchestrator only stamps
                # the walk with its own clock, which it alone owns.
                process_groups = self._process_groups.sample_if_due(stat_path)
                if process_groups is not None:
                    process_groups["at_s"] = (
                        time.monotonic() - self._started_at
                        if self._started_at is not None else elapsed_s
                    )

        if isinstance(container_name, str) and container_name.strip():
            stats = collect_container_stats(runtime, container_name.strip())
            if stats is not None:
                cpu_percent = stats.get("cpu_percent")
                if cpu_time_ns is None:  # cgroup unavailable: fall back
                    cpu_time_ns = stats.get("cpu_time_ns")
                rss_bytes = stats.get("rss_bytes")
                memory_limit_bytes = stats.get("memory_limit_bytes")
                num_pids = stats.get("num_pids")
                block_read_bytes = stats.get("block_read_bytes")
                block_write_bytes = stats.get("block_write_bytes")
                status = stats.get("status", status)
            else:
                status = inspect_container_status(runtime, container_name.strip())

        # Live frame index, sampled every tick. Needed to tell a run that is
        # STALLED under a cap from one that is merely SLOW: a timed-out cell may
        # only be called stall-bound when the cgroup shows reclaim activity AND
        # frames stopped advancing. Without the time series, a short timeout
        # silently converts "slow" into "bound".
        deadline_frame = self._sample_progress_frame()

        gpu_sample = self._gpu_telemetry.read()

        # Split GPU use between the SLAM's own control group and everything
        # else on the card (a GPU load generator, or anything the host is
        # running). NVML reports host PIDs, the same namespace as cgroup.procs.
        gpu_processes = None
        per_pid = self._gpu_processes.read()
        if per_pid:
            slam_pids = set()
            if stat_path is not None:
                try:
                    slam_pids = {
                        int(x) for x in (stat_path.parent / "cgroup.procs").read_text().split()
                    }
                except Exception:
                    slam_pids = set()
            groups = {
                "slam": {"vram_mb": 0, "sm_util": 0, "procs": 0},
                "other": {"vram_mb": 0, "sm_util": 0, "procs": 0},
            }
            for pid, vals in per_pid.items():
                g = groups["slam" if pid in slam_pids else "other"]
                g["procs"] += 1
                if vals.get("vram_mb") is not None:
                    g["vram_mb"] += vals["vram_mb"]
                if vals.get("sm_util") is not None:
                    g["sm_util"] += vals["sm_util"]
            gpu_processes = groups

        # Sample live load antagonists too, so every cell carries per-phase
        # evidence that contention was actually sustained (spawn-time
        # verification alone cannot show an antagonist that died or
        # throttled mid-phase).
        antagonist_samples: Optional[Dict[str, Any]] = None
        if self._load_controller is not None:
            names = self._load_controller.active_container_names()
            if names:
                antagonist_samples = {}
                for antagonist_name in names:
                    a_stats = collect_container_stats(runtime, antagonist_name)
                    # Stamp the instant this counter was read, for the same
                    # reason the SLAM's counter is stamped: these reads happen
                    # late in a sample, where assembly latency is worst, and a
                    # rate computed against the sample's own timestamp charges
                    # that latency to the interval.
                    a_read_at = (
                        time.monotonic() - self._started_at
                        if self._started_at is not None else elapsed_s
                    )
                    if a_stats is not None:
                        antagonist_samples[antagonist_name] = {
                            "cpu_percent": a_stats.get("cpu_percent"),
                            "cpu_time_ns": a_stats.get("cpu_time_ns"),
                            "cpu_time_at_s": a_read_at,
                            "rss_bytes": a_stats.get("rss_bytes"),
                            "num_pids": a_stats.get("num_pids"),
                            "status": a_stats.get("status"),
                        }
                    else:
                        antagonist_samples[antagonist_name] = {"status": "unavailable"}

            # Antagonists that are NOT their own container: the in-container
            # crowd (runs inside the SLAM's cgroup) and the segmentation
            # co-tenant (a host process). Neither appears in
            # active_container_names(), so before this both were verified once
            # at spawn and then invisible -- a crowd dying at minute two left the
            # SLAM uncontended while the cell still reported as contended.
            # Throttled inside the controller, since counting PIDs costs a
            # `podman exec` that competes with the workload being measured.
            try:
                liveness = self._load_controller.process_antagonist_liveness()
            except Exception:
                # Telemetry must never take down a run. An unavailable liveness
                # read is itself recorded below rather than raised.
                liveness = {"liveness_error": True}
            if liveness:
                antagonist_samples = antagonist_samples or {}
                antagonist_samples.update(liveness)

        self.trace.append(
            {
                "elapsed_s": round(elapsed_s, 6),
                "phase": phase_name,
                "target_kind": self._target_kind,
                "container_name": container_name,
                "process_alive": process_alive,
                "process_cpu_percent": None,
                "container_cpu_percent": cpu_percent,
                "container_cpu_time_ns": cpu_time_ns,
                "container_cpu_time_at_s": cpu_time_at_s,
                "container_cpu_throttling": throttling or None,
                "container_memory_events": memory_events or None,
                "deadline_frame": deadline_frame,
                "process_groups": process_groups,
                "system_cpu_percent": psutil.cpu_percent(None),
                "rss_bytes": rss_bytes,
                "memory_limit_bytes": memory_limit_bytes,
                "num_threads": None,
                "num_pids": num_pids,
                "status": status,
                "container_io_read_bytes_total": block_read_bytes,
                "container_io_write_bytes_total": block_write_bytes,
                "gpu_util_percent": gpu_sample.get("gpu_util_percent"),
                "gpu_mem_used_mb": gpu_sample.get("gpu_mem_used_mb"),
                "gpu_mem_total_mb": gpu_sample.get("gpu_mem_total_mb"),
                "gpu_power_w": gpu_sample.get("gpu_power_w"),
                "gpu_temp_c": gpu_sample.get("gpu_temp_c"),
                "gpu_sm_clock_mhz": gpu_sample.get("gpu_sm_clock_mhz"),
                "gpu_processes": gpu_processes,
                "load_antagonists": antagonist_samples,
            }
        )

    def _install_log_capture(self) -> None:
        """Attach the capture to this package's logger for the life of the run.

        Attached at ``slamadversariallab.runtime_stress`` rather than per module,
        so every controller's output is caught through normal propagation with no
        call-site changes.
        """
        if self._log_capture is not None:
            return
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            capture = _HarnessLogCapture(
                self.output_dir / "harness.log", self.events, self._elapsed_s
            )
        except Exception as exc:
            # Best-effort: losing the audit trail is bad, losing the run is worse.
            logger.warning("Could not open harness.log for this run: %s", exc)
            return
        package_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
        package_logger.addHandler(capture)
        # A parent that filters at WARNING would drop the INFO records this is
        # here to keep. Only ever lower the threshold, never raise it.
        if package_logger.level == logging.NOTSET or package_logger.level > logging.INFO:
            package_logger.setLevel(logging.INFO)
        self._log_capture = capture

    def _remove_log_capture(self) -> None:
        """Detach and close. A campaign runs many runs in one process, so a
        leaked handler would keep writing every later run into the first run's
        file and append its warnings to a stale events list."""
        capture = self._log_capture
        if capture is None:
            return
        self._log_capture = None
        try:
            logging.getLogger(__name__.rsplit(".", 1)[0]).removeHandler(capture)
        except Exception:
            pass
        try:
            capture.close()
        except Exception:
            pass

    def _record_event(
        self, kind: str, elapsed_s: float, message: str, frame: Optional[int] = None
    ) -> None:
        event: Dict[str, Any] = {
            "kind": kind,
            "elapsed_s": round(elapsed_s, 6),
            "message": message,
        }
        if frame is not None:
            event["frame"] = frame
        self.events.append(event)

    def _terminate_target(self) -> None:
        """Terminate the attached execution target."""
        if self._target_kind in CONTAINER_TARGET_KINDS:
            runtime = _container_runtime_binary(self._target_kind)
            container_name = self._target_metadata.get("container_name")
            if isinstance(container_name, str) and container_name.strip():
                try:
                    subprocess.run(
                        [runtime, "stop", "-t", "5", container_name.strip()],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                except Exception:
                    pass
                try:
                    subprocess.run(
                        [runtime, "rm", "-f", container_name.strip()],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=10,
                    )
                except Exception:
                    pass
            return

        process = self._process
        if process is None or process.poll() is not None:
            return

        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
