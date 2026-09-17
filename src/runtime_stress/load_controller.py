"""Load-antagonist controller: contention-based runtime stress.

Where the cap controllers SHRINK the SLAM's own allocation (``podman update``
on the target container), this controller ADDS calibrated load.

Supported paths:
- ``in_container``: stress-ng workers exec'd INTO the SLAM's own cgroup
  (flat, equal-priority competition). This is the supported CPU-side mode.
- ``gpu``: the GPU antagonist sibling container (VRAM ballast + duty-cycled
  matmul). Sibling by necessity -- it is the only GPU-load mechanism.

DEPRECATED path (kept working, not removed):
- the sibling stress-ng container (``cpu_workers``/``stream_workers``/
  ``vm_workers`` + ``fence`` in quota or weight mode). Conditions
  C8/C9/C11/C12/C14/C15 were measured with it and those results are published,
  so it stays reproducible; new scenarios should use ``in_container``. A
  warning fires once per run when the deprecated path is used.

The sibling stress-ng container adds calibrated load beside the SLAM:

- one stress-ng container (official image) for CPU-side antagonists
  (``--cpu`` hogs, ``--stream`` DRAM/L3 bandwidth pollution, ``--vm``
  allocation churn), fenced with podman caps so the pressure is exact;
- one GPU antagonist container (``gpu_antagonist.py`` bind-mounted into a
  torch image) producing VRAM ballast + duty-cycled matmul contention,
  spawned once in ``prepare()`` (CUDA init overlaps the SLAM's model load)
  and reconfigured per phase through an epoch-stamped control file.

Fail-loud contract: every spawn/reconfigure is verified (container Running,
worker count via ``podman top``, fence readback, status-file epoch ack); any
failure raises so the orchestrator records controller_error and terminates
the run — a cell must never be scored as contended when it was not.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from .controllers import ResourceController
from .models import LoadControl, LoadGpuAntagonist, LoadInContainer, RuntimeStressControls

logger = logging.getLogger(__name__)

STRESS_NG_IMAGE = "ghcr.io/colinianking/stress-ng:latest"
# Repo root, derived once. The segmentation co-tenant is launched with this as
# its cwd so `python -m src.runtime_stress.sam3_antagonist` resolves regardless
# of where the campaign was invoked from.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Static stress-ng binary bind-mounted into SLAM containers for in-container
# load (extracted from the official image; see deps/stress-ng/extract_static.sh).
STATIC_STRESS_NG_HOST_PATH = (
    REPO_ROOT / "deps" / "stress-ng" / "stress-ng-static"
)
STRESS_NG_CONTAINER_PATH = "/sal/stress-ng"
DEFAULT_GPU_IMAGE = "localhost/droidslam:latest"
# The GPU antagonist's own host-side footprint is pinned by constants so it
# can never become a confound (it is not a config knob in v1).
GPU_FENCE_CPUS = "2"
GPU_FENCE_MEMORY = "8g"
LOAD_LABEL = "sal.load"
CPU_METHOD = "double"  # pure-FP tiny-working-set: orthogonal to --stream

_CONTAINER_SCRIPT = "/sal/gpu_antagonist.py"
_CONTAINER_LOAD_DIR = "/sal/load"


class PodmanLoadController(ResourceController):
    """Spawn and manage fenced antagonist containers for one scenario run."""

    def __init__(
        self,
        load_specs: Optional[List[LoadControl]] = None,
        attach_timeout_s: float = 15.0,
        gpu_ready_timeout_s: float = 90.0,
        gpu_ack_timeout_s: float = 10.0,
    ) -> None:
        self._load_specs = list(load_specs or [])
        self._attach_timeout_s = attach_timeout_s
        self._gpu_ready_timeout_s = gpu_ready_timeout_s
        self._gpu_ack_timeout_s = gpu_ack_timeout_s

        # Generated at construction (not prepare) so prewarm_gpu() can name
        # and label containers before the SLAM target exists.
        self._session_id: Optional[str] = uuid.uuid4().hex[:8]
        self._target_container: Optional[str] = None
        self._sng_container: Optional[str] = None
        self._sng_spec: Optional[LoadControl] = None
        self._gpu_container: Optional[str] = None
        self._gpu_spec: Optional[LoadGpuAntagonist] = None
        self._gpu_image: Optional[str] = None
        self._seg_proc = None                    # SAM 3 co-tenant subprocess
        # Previous (cpu_ticks, wall_clock) for the co-tenant's process tree, so
        # cores can be reported as a RATE rather than a cumulative total.
        self._seg_cpu_prev: Optional[tuple] = None
        self._seg_spec = None
        self._seg_status_path: Optional[Path] = None
        self._seg_ready_elapsed_s: Optional[float] = None
        self._seg_paused: bool = False
        self._liveness_cache: Optional[Dict[str, Any]] = None
        self._liveness_checked_at: float = 0.0
        self._gpu_dir: Optional[Path] = None
        self._gpu_epoch = 0
        self._gpu_ready_elapsed_s: Optional[float] = None
        self._image_digests: Dict[str, str] = {}
        self._sibling_deprecation_warned = False
        # In-container stress-ng (runs inside the SLAM's own cgroup).
        self._incontainer_active = False
        self._incontainer_last_spec: Optional[LoadInContainer] = None
        self._incontainer_stop = f"/tmp/sal-burn-stop-{self._session_id}"

    # ------------------------------------------------------------------
    # ResourceController interface (+ prewarm hold point)
    # ------------------------------------------------------------------

    def prewarm_gpu(self) -> bool:
        """Spawn and ready the GPU antagonist BEFORE the SLAM launches.

        The hold point: the antagonist's CUDA init (5-15 s) previously
        overlapped the SLAM's startup, leaving fast-starting SLAMs an
        uncontended head at the beginning of the stress phase. Called from
        the orchestrator before the SLAM container is spawned, so contention
        capability exists from the SLAM's very first frame. Returns True if
        a GPU antagonist was specced and is now ready, False if the specs
        request no GPU antagonist. Raises (fail loud) on any spawn/ready
        failure — before a SLAM exists, so nothing can be mis-scored.
        """
        gpu_spec = next((s.gpu for s in self._load_specs if s.gpu is not None), None)
        if gpu_spec is None:
            return False
        if self._gpu_container is not None:
            return True  # already prewarmed
        self._gpu_spec = gpu_spec
        self._gpu_image = gpu_spec.image or DEFAULT_GPU_IMAGE
        self._require_image(
            self._gpu_image,
            hint="build or pull a torch-capable image and set "
            "controls.load.gpu.image if not using the default",
        )
        self._spawn_gpu_antagonist()
        return True

    def prewarm_segmentation(self) -> bool:
        """Ready the SAM 3 co-tenant BEFORE the SLAM launches.

        Same hold point as `prewarm_gpu`, and for the same reason its docstring
        gives. SAM 3 takes ~9 s to load its weights and complete a warm
        inference. Spawning it when the stress PHASE begins meant the SLAM was
        already running through that load, so the run carried an uncontended
        head: measured 26% coverage on dpvslam (4.4 s of contention in a 16.8 s
        run) and 50% on droidslam. Every degradation measured that way is a lower
        bound, and worse, coverage varied 26-80% across systems, so severities
        were not comparable between rows.

        Unlike the GPU antagonist this needs no container and no SLAM, so it can
        be readied arbitrarily early. Returns True if a segmentation spec exists
        and is now ready, False if none is requested. Raises before a SLAM exists,
        so nothing can be mis-scored.
        """
        spec = next((s.segmentation for s in self._load_specs
                     if s.segmentation is not None), None)
        if spec is None:
            return False
        if self._seg_proc is not None:
            return True  # already prewarmed
        self._spawn_segmentation(spec)
        return True

    def prepare(self, process, *, target_kind: str, target_metadata=None) -> None:
        if target_kind != "podman_container":
            raise RuntimeError(
                "Load-antagonist runtime stress is only supported for "
                f"podman_container targets in v1, got '{target_kind}'."
            )
        metadata = dict(target_metadata or {})
        self._target_container = metadata.get("container_name")
        if not self._target_container:
            raise RuntimeError(
                "Load-antagonist controller requires target_metadata['container_name'] "
                "(needed to protect the SLAM container from antagonist teardown)."
            )

        # Await the TARGET container's existence before any failable work
        # (house pattern, cf. PodmanCpuController.prepare). `podman run`
        # creates the container asynchronously; if prepare fails before it
        # exists, the orchestrator's terminate-on-prepare-failure hits
        # "no such container" and the SLAM survives as an orphan, finishing
        # as a valid-looking UNstressed run.
        self._await_target_exists(self._target_container, self._attach_timeout_s)

        needs_stress_ng = any(self._spec_has_stress_ng(s) for s in self._load_specs)
        if needs_stress_ng:
            self._require_image(
                STRESS_NG_IMAGE, hint=f"podman pull {STRESS_NG_IMAGE}"
            )
        # GPU antagonist: normally already up via prewarm_gpu() (the hold
        # point). Spawning here is the fallback for callers that attach
        # without prewarming; idempotent when prewarmed.
        if self._gpu_container is None:
            self.prewarm_gpu()

    def apply(self, controls: RuntimeStressControls) -> None:
        load = controls.load
        # EVERY load kind must appear in this guard. A kind missing from it is
        # torn down and returned past before its own dispatch branch below is
        # ever reached -- the phase then runs with NO antagonist while the cell
        # still reports as stressed. That happened to `segmentation`: the branch
        # was written, the guard was not updated, and the smoke run produced
        # `load_antagonists: []` with no error anywhere.
        if load is None or (
            not self._spec_has_stress_ng(load)
            and load.gpu is None
            and load.in_container is None
            and load.segmentation is None
        ):
            # Mirror the cap controllers' "control absent -> release" idiom:
            # phase without load = no contention.
            self._teardown_stress_ng()
            self._idle_gpu()
            self._teardown_in_container()
            # IDLE, not teardown: destroying it here undoes the prewarm and
            # makes the next phase pay the full model load while the SLAM runs.
            self._idle_segmentation()
            return

        if self._spec_has_stress_ng(load):
            # Kill-and-respawn per phase: stress-ng starts in <1 s and a
            # respawn is simpler and more verifiable than diffing a spec.
            self._teardown_stress_ng()
            self._spawn_stress_ng(load)
        else:
            self._teardown_stress_ng()

        if load.gpu is not None:
            if self._gpu_container is None:
                raise RuntimeError(
                    "Phase requests a GPU antagonist but none was spawned in "
                    "prepare() — scenario phases disagree with the compiled "
                    "load specs (this is a bug, not a config error)."
                )
            self._reconfigure_gpu(load.gpu)
        else:
            self._idle_gpu()

        if load.in_container is not None:
            self._teardown_in_container()  # fresh sentinel per phase
            self._spawn_in_container(load.in_container)
        else:
            self._teardown_in_container()

        if load.segmentation is not None:
            # Idempotent across phases: SAM 3 costs ~9 s to load, so a phase
            # boundary must not pay that again. Resume a prewarmed-then-idled
            # co-tenant; respawn only if it is absent or the spec moved.
            if self._seg_proc is not None and self._seg_spec == load.segmentation:
                self._resume_segmentation()
            else:
                self._teardown_segmentation()
                self._spawn_segmentation(load.segmentation)
        else:
            self._idle_segmentation()

    def release(self) -> None:
        self._teardown_stress_ng()
        self._teardown_gpu()
        self._teardown_in_container()
        self._teardown_segmentation()

    def cleanup(self) -> None:
        try:
            self._teardown_stress_ng()
        except Exception:
            pass
        try:
            self._teardown_gpu()
        except Exception:
            pass
        try:
            self._teardown_in_container()
        except Exception:
            pass
        # Label-scoped backstop sweep: catches anything tracking lost.
        if self._session_id:
            try:
                result = self._run(
                    ["podman", "ps", "-aq", "--filter",
                     f"label={LOAD_LABEL}.session={self._session_id}"],
                    timeout=10,
                )
                for cid in (result.stdout or "").split():
                    if self._target_container and cid.startswith(self._target_container):
                        continue  # defensive: never touch the SLAM container
                    self._run(["podman", "rm", "-f", cid], timeout=15)
            except Exception:
                pass
        if self._gpu_dir is not None:
            shutil.rmtree(self._gpu_dir, ignore_errors=True)
            self._gpu_dir = None
        self._session_id = None

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def active_container_names(self) -> List[str]:
        """Names of antagonist containers currently expected to be live.

        Consumed by the orchestrator's telemetry loop so per-phase antagonist
        stats land in stress_trace.json alongside the SLAM's own samples.
        """
        return [n for n in (self._sng_container, self._gpu_container) if n]

    # How often liveness is re-checked for the non-container antagonists.
    # Deliberately slower than the telemetry tick: counting PIDs inside the
    # SLAM's container costs a `podman exec`, and that exec competes with the
    # very workload being measured. We are detecting DEATH, not tracking a
    # fast-moving signal, so seconds of resolution is ample.
    LIVENESS_MIN_INTERVAL_S = 5.0

    def process_antagonist_liveness(self) -> Dict[str, Any]:
        """Liveness for antagonists that are NOT their own container.

        `active_container_names()` covers the sibling stress-ng and GPU
        antagonists, which the orchestrator samples via container stats. Two
        kinds fall outside it and were therefore invisible for their whole run:

          in_container  runs INSIDE the SLAM's cgroup, so it has no container of
                        its own to sample.
          segmentation  is a host process, not a container at all.

        Both were verified once at spawn and never again. That leaves the
        failure this axis cannot survive undetectable: a crowd that dies at
        minute two leaves the SLAM running uncontended for the rest of the cell,
        which then reports as contended and looks like the system SURVIVED
        contention. The delivered-dose average is only a partial defence, since a
        crowd dying late in a long run can still land inside tolerance.

        Returns {} when nothing applies, so the caller can merge unconditionally.
        Results are cached for LIVENESS_MIN_INTERVAL_S.
        """
        now = time.monotonic()
        if (self._liveness_cache is not None
                and now - self._liveness_checked_at < self.LIVENESS_MIN_INTERVAL_S):
            return self._liveness_cache

        out: Dict[str, Any] = {}

        if self._incontainer_active and self._target_container:
            spec = self._incontainer_last_spec
            expected = 0
            if spec is not None:
                expected = (spec.cpu_workers or 0) + (spec.stream_workers or 0) \
                    + (spec.vm_workers or 0)
            alive = None
            probe = self._run(
                ["podman", "exec", self._target_container,
                 "sh", "-c", f"pgrep -c -f {STRESS_NG_CONTAINER_PATH} || true"],
                timeout=10,
            )
            if probe.returncode == 0:
                try:
                    alive = int((probe.stdout or "0").strip().splitlines()[0])
                except (ValueError, IndexError):
                    alive = None
            out["in_container"] = {
                "workers_alive": alive,
                "workers_expected": expected,
                # The supervisor process counts too, so alive is expected to be
                # workers+1. Flagging on `alive <= 1` catches total death without
                # firing on the ordinary off-by-one.
                "died": (alive is not None and expected > 0 and alive <= 1),
            }

        if self._seg_proc is not None:
            status = self._read_json(self._seg_status_path) or {}
            rc = self._seg_proc.poll()
            out["segmentation"] = {
                "alive": rc is None,
                "returncode": rc,
                "state": status.get("state"),
                "frames_done": status.get("frames_done"),
                "passes": status.get("passes"),
                "fps": status.get("fps"),
                **self._seg_cpu_sample(),
            }

        self._liveness_cache = out
        self._liveness_checked_at = now
        return out

    def _seg_cpu_sample(self) -> Dict[str, Any]:
        """Mean cores and RSS for the co-tenant's whole process tree.

        SAM 3 is a HOST PROCESS, not a container, so container stats never see
        it and the per-workload CPU channel does not either -- that channel
        covers the SLAM's cgroup, and its `load` group reads zero procs in every
        co-tenant run. Without this the paper could say what the co-tenant costs
        on the GPU and nothing about what it costs on the processor.

        The tree matters, not just the leader. It is spawned with
        `start_new_session=True`, so it is a session leader and every descendant
        shares its session id -- field 6 of /proc/<pid>/stat. Summing by session
        catches worker processes that summing the leader alone would miss.

        Cores is a RATE: ticks consumed since the previous call over wall time
        elapsed, so 1.0 means one processor fully busy. Returns None on the first
        call, because a rate needs two samples.
        """
        if self._seg_proc is None:
            return {}
        # Narrow guard, not a blanket try: a handle with no pid cannot be
        # sampled and that is a fact about the handle, not an error to swallow.
        # Liveness reporting uses doubles that carry only poll(), and telemetry
        # must never be the thing that kills a run.
        leader = getattr(self._seg_proc, "pid", None)
        if leader is None:
            return {}
        ticks, rss_pages, procs = 0, 0, 0
        try:
            hz = os.sysconf("SC_CLK_TCK") or 100
            page = os.sysconf("SC_PAGE_SIZE") or 4096
        except (ValueError, OSError):
            hz, page = 100, 4096
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat") as f:
                    raw = f.read()
            except OSError:
                continue        # exited between listdir and open; normal
            # comm can contain spaces and parentheses, so split after the LAST
            # ')' rather than tokenising the whole line.
            close = raw.rfind(")")
            if close < 0:
                continue
            fields = raw[close + 2:].split()
            # fields[0] is state, so session (field 6) is index 3 here.
            if len(fields) < 22:
                continue
            try:
                if int(fields[3]) != leader:
                    continue
                ticks += int(fields[11]) + int(fields[12])   # utime + stime
                rss_pages += int(fields[21])
            except (ValueError, IndexError):
                continue
            procs += 1
        now = time.monotonic()
        cores = None
        if self._seg_cpu_prev is not None:
            prev_ticks, prev_at = self._seg_cpu_prev
            dt = now - prev_at
            if dt > 0 and ticks >= prev_ticks:
                cores = round(((ticks - prev_ticks) / hz) / dt, 3)
        self._seg_cpu_prev = (ticks, now)
        return {"cpu_cores": cores, "rss_mb": round(rss_pages * page / 1e6, 1),
                "procs": procs}

    def antagonist_summaries(self) -> List[Dict[str, Any]]:
        """Per-antagonist records for stress_summary.json (finalize snapshot)."""
        out: List[Dict[str, Any]] = []
        if self._sng_container is not None and self._sng_spec is not None:
            s = self._sng_spec
            out.append(
                {
                    "kind": "stress_ng",
                    "container_name": self._sng_container,
                    "image": STRESS_NG_IMAGE,
                    "image_digest": self._image_digests.get(STRESS_NG_IMAGE),
                    "cpu_workers": s.cpu_workers,
                    "stream_workers": s.stream_workers,
                    "vm_workers": s.vm_workers,
                    "vm_bytes_mb": s.vm_bytes_mb,
                    "fence": {
                        "cpus": s.fence.cpus if s.fence else None,
                        "memory_mb": s.fence.memory_mb if s.fence else None,
                        "cpu_shares": s.fence.cpu_shares if s.fence else None,
                    },
                }
            )
        if self._gpu_container is not None and self._gpu_spec is not None:
            out.append(
                {
                    "kind": "gpu",
                    "container_name": self._gpu_container,
                    "image": self._gpu_image,
                    "image_digest": self._image_digests.get(self._gpu_image or ""),
                    "vram_mb": self._gpu_spec.vram_mb,
                    "matmul_n": self._gpu_spec.matmul_n,
                    "duty_cycle": self._gpu_spec.duty_cycle,
                    "fence": {"cpus": float(GPU_FENCE_CPUS), "memory": GPU_FENCE_MEMORY},
                    "ready_elapsed_s": self._gpu_ready_elapsed_s,
                }
            )
        if self._incontainer_last_spec is not None:
            ic = self._incontainer_last_spec
            out.append(
                {
                    "kind": "in_container",
                    "container_name": self._target_container,
                    "generator": "stress-ng (static, bind-mounted)",
                    "cpu_workers": ic.cpu_workers,
                    "stream_workers": ic.stream_workers,
                    "vm_workers": ic.vm_workers,
                    "vm_bytes_mb": ic.vm_bytes_mb,
                    "note": "runs in the SLAM's cgroup; container_cpu_percent in "
                            "the trace is COMBINED (SLAM+load), so it does not "
                            "separate them. The deadline drop rate is the exact "
                            "SLAM-health metric.",
                }
            )
        if self._seg_spec is not None:
            spec = self._seg_spec
            # Final status carries frames_done and passes. Reading it here means
            # a co-tenant that died early, or never wrapped the sequence, is
            # VISIBLE in the run record rather than assumed to have run
            # throughout -- the difference between a contended cell and one that
            # only looked contended.
            status = self._read_json(self._seg_status_path) or {}
            out.append(
                {
                    "kind": "segmentation",
                    "model": "SAM 3",
                    "conda_env": spec.conda_env,
                    "frames_dir": spec.frames_dir,
                    "prompt": spec.prompt,
                    "max_frames": spec.max_frames,
                    "ready_elapsed_s": self._seg_ready_elapsed_s,
                    "state": status.get("state"),
                    "frames_done": status.get("frames_done"),
                    "passes": status.get("passes"),
                    "fps": status.get("fps"),
                    "frames_in_sequence": status.get("frames_in_sequence"),
                    "note": "sibling PROCESS in its own conda env (SAM 3 needs "
                            "torch 2.10/cu128, the SLAMs are cu118). Holds "
                            "88-100% SM alone, so it is a saturating GPU "
                            "co-tenant rather than one rung of a ladder. It "
                            "shares the dataset's images and frame budget only "
                            "-- NOT the SLAM's deadline, pacing or phases.",
                }
            )
        return out

    # ------------------------------------------------------------------
    # SAM 3 segmentation co-tenant
    # ------------------------------------------------------------------

    SEG_READY_TIMEOUT_S = 180.0

    def _spawn_segmentation(self, spec) -> None:
        """Start SAM 3 beside the SLAM and BLOCK until it is actually working.

        Unlike stress-ng, this antagonist is not contending a millisecond after
        exec: it loads 3.5 GB of weights and pays CUDA autotuning on its first
        frame, about 6 s in total. A phase that began during that window would
        measure an unstressed prefix and report it as stressed, so the phase does
        not start until the runner has completed one inference and written
        `state: ready`.

        Runs in its own conda env because SAM 3 needs torch 2.10/cu128 while the
        SLAMs are on cu118. It is a sibling PROCESS rather than a container --
        the GPU is shared either way, and a process avoids building an image for
        a workload that only needs to sit next to the SLAM.
        """
        frames_dir = Path(spec.frames_dir)
        if not frames_dir.is_dir():
            raise RuntimeError(
                f"Segmentation co-tenant frames_dir does not exist: {frames_dir}. "
                f"Refusing to run a cell whose antagonist would segment nothing "
                f"and leave the run looking stressed while it was not."
            )

        status_dir = Path(tempfile.mkdtemp(prefix="sam3_antagonist_"))
        self._seg_status_path = status_dir / "status.json"
        module = "src.runtime_stress.sam3_antagonist"
        # Call the ENVIRONMENT'S python directly rather than going through
        # `conda run`. conda run is a WRAPPER: Popen returns the wrapper's pid,
        # so terminating it kills the wrapper and ORPHANS the python underneath.
        # Measured: 4 processes before teardown, 3 after, with the real worker
        # still holding ~95% of the GPU. A leaked co-tenant then contends with
        # every later cell while those cells record no antagonist at all --
        # an invisible confound in exactly the direction that fakes a result.
        env_python = Path.home() / "miniconda3" / "envs" / spec.conda_env / "bin" / "python"
        if not env_python.exists():
            raise RuntimeError(
                f"Segmentation co-tenant needs the '{spec.conda_env}' conda env "
                f"but {env_python} does not exist. See "
                f"the SAM 3 upstream README for the environment recipe."
            )
        cmd = [
            str(env_python), "-m", module,
            "--frames-dir", str(frames_dir),
            "--status", str(self._seg_status_path),
            "--prompt", spec.prompt,
        ]
        if spec.max_frames:
            cmd += ["--max-frames", str(spec.max_frames)]

        started = time.monotonic()
        # Own process group, so teardown can signal the worker AND anything it
        # spawns in one call.
        self._seg_proc = subprocess.Popen(
            cmd, cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )
        self._seg_spec = spec

        # Poll the status file rather than sleeping a fixed time: load cost
        # varies with disk cache and GPU state, and a fixed sleep would either
        # waste time or start the phase early.
        deadline = started + self.SEG_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._seg_proc.poll() is not None:
                out = (self._seg_proc.stdout.read() if self._seg_proc.stdout else "")[-800:]
                raise RuntimeError(
                    f"Segmentation co-tenant exited before becoming ready "
                    f"(rc={self._seg_proc.returncode}). Output:\n{out}"
                )
            status = self._read_json(self._seg_status_path)
            if status:
                if status.get("state") == "error":
                    raise RuntimeError(
                        f"Segmentation co-tenant failed: {status.get('message')}"
                    )
                if status.get("state") in ("ready", "running"):
                    self._seg_ready_elapsed_s = round(time.monotonic() - started, 2)
                    logger.info(
                        "  Segmentation co-tenant ready in %.1fs (SAM 3, prompt=%r, "
                        "%s frames)",
                        self._seg_ready_elapsed_s, spec.prompt,
                        status.get("frames_in_sequence"),
                    )
                    return
            time.sleep(0.5)

        self._teardown_segmentation()
        raise RuntimeError(
            f"Segmentation co-tenant did not report ready within "
            f"{self.SEG_READY_TIMEOUT_S:.0f}s. Refusing to run the phase "
            f"uncontended while reporting it as loaded."
        )

    def _idle_segmentation(self) -> None:
        """Pause the co-tenant WITHOUT unloading it, for a phase that wants no load.

        The GPU antagonist survives a no-load phase by reconfiguring its duty to
        zero -- its container stays up, so the next phase costs nothing to
        re-enter. Segmentation used teardown instead, which killed the process
        outright. That silently undid the prewarm: the warmup phase (no load)
        destroyed the co-tenant readied before the SLAM launched, and the stress
        phase paid the ~9 s load again while the SLAM ran uncontended. Measured
        coverage stayed at 27% on dpvslam even WITH prewarm, and the log showed
        the co-tenant becoming ready twice in one run.

        SIGSTOP is the right idle here because this antagonist has no dose to
        turn down. A stopped process is not scheduled, so it does no CUDA work,
        while its context and 3.5 GB of weights stay resident -- which is what
        makes resuming instant instead of another 9 s.
        """
        proc = self._seg_proc
        if proc is None or proc.poll() is not None or self._seg_paused:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGSTOP)
            self._seg_paused = True
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("  Segmentation co-tenant idle: %s", exc)

    def _resume_segmentation(self) -> None:
        """Un-pause a co-tenant idled for a previous phase."""
        proc = self._seg_proc
        if proc is None or proc.poll() is not None or not self._seg_paused:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
            self._seg_paused = False
        except (ProcessLookupError, PermissionError) as exc:
            logger.warning("  Segmentation co-tenant resume: %s", exc)

    def _teardown_segmentation(self) -> None:
        """Stop the co-tenant, preferring a clean SIGTERM so it writes a final
        status (frames done, passes over the sequence) before exiting."""
        proc = self._seg_proc
        if proc is None:
            return
        try:
            if proc.poll() is None:
                # A SIGSTOPped process never handles SIGTERM. Resume first so it
                # can write its final status instead of needing SIGKILL.
                if self._seg_paused:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
                        self._seg_paused = False
                    except (ProcessLookupError, PermissionError):
                        pass
                # Signal the whole GROUP, not just the leader. Killing only the
                # leader is what leaked the worker when this went through
                # `conda run`, and it would leak any child the worker spawns.
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        proc.kill()
                    proc.wait(timeout=10)
        except Exception as exc:  # teardown must never mask the run's own result
            logger.warning("  Segmentation co-tenant teardown: %s", exc)
        finally:
            self._seg_proc = None

    # ------------------------------------------------------------------
    # stress-ng antagonist
    # ------------------------------------------------------------------

    @staticmethod
    def _spec_has_stress_ng(spec: LoadControl) -> bool:
        return bool(spec.cpu_workers or spec.stream_workers or spec.vm_workers)

    def _spawn_stress_ng(self, load: LoadControl) -> None:
        if not self._sibling_deprecation_warned:
            self._sibling_deprecation_warned = True
            logger.warning(
                "  DEPRECATED stressor: the sibling stress-ng antagonist "
                "(cpu_workers/stream_workers/vm_workers + fence) is deprecated. "
                "Use controls.load.in_container for CPU-side contention. This "
                "path still runs, so earlier results measured with it remain reproducible."
            )
        if load.fence is None or (
            load.fence.cpus is None and load.fence.cpu_shares is None
        ):
            raise RuntimeError(
                "stress-ng antagonist requires fence.cpus or fence.cpu_shares "
                "(schema validation should have caught this)."
            )
        name = f"sal-load-sng-{self._session_id}"
        cmd = [
            "podman", "run", "-d", "--rm",
            "--name", name,
            "--label", f"{LOAD_LABEL}=1",
            "--label", f"{LOAD_LABEL}.session={self._session_id}",
        ]
        # Quota mode: cap the antagonist (SLAM stays protected by fair share).
        if load.fence.cpus is not None:
            cmd.extend(["--cpus", f"{load.fence.cpus:g}"])
        # Weight mode: run UNCAPPED but high-priority, collapsing the SLAM's
        # proportional fair-share slice (models a best-effort SLAM under load).
        if load.fence.cpu_shares is not None:
            cmd.extend(["--cpu-shares", str(load.fence.cpu_shares)])
        if load.fence.memory_mb is not None:
            cmd.extend(["--memory", f"{load.fence.memory_mb}m"])
        cmd.append(STRESS_NG_IMAGE)
        total_workers = 0
        if load.cpu_workers:
            cmd.extend(["--cpu", str(load.cpu_workers), "--cpu-method", CPU_METHOD])
            total_workers += load.cpu_workers
        if load.stream_workers:
            cmd.extend(["--stream", str(load.stream_workers)])
            total_workers += load.stream_workers
        if load.vm_workers:
            cmd.extend(["--vm", str(load.vm_workers), "--vm-bytes", f"{load.vm_bytes_mb}M"])
            total_workers += load.vm_workers

        result = self._run(cmd, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to spawn stress-ng antagonist: {result.stderr.strip()[:300]}"
            )
        self._sng_container = name
        self._sng_spec = load

        self._await_running(name, self._attach_timeout_s)
        self._verify_worker_count(name, total_workers)
        if load.fence.cpus is not None:
            self._verify_fence_cpus(name, load.fence.cpus)
        if load.fence.cpu_shares is not None:
            self._verify_fence_shares(name, load.fence.cpu_shares)
        self._record_digest(STRESS_NG_IMAGE)
        if load.fence.cpus is not None:
            fence_desc = f"quota {load.fence.cpus:g}c"
        else:
            fence_desc = f"weight {load.fence.cpu_shares} (uncapped)"
        logger.info(
            "  Load antagonist up: %s (%d workers, %s%s)",
            name,
            total_workers,
            fence_desc,
            f", {load.fence.memory_mb}MB" if load.fence.memory_mb else "",
        )

    def _teardown_stress_ng(self) -> None:
        if self._sng_container is None:
            return
        self._rm_container(self._sng_container)
        self._sng_container = None
        self._sng_spec = None

    def _verify_worker_count(
        self, name: str, expected_workers: int, timeout_s: float = 5.0
    ) -> None:
        """Poll until the requested worker count is visible (fail loud on timeout).

        stress-ng forks its workers shortly after start; a single-shot check
        can land inside that fork window and see only the supervisor, so the
        count is polled briefly before declaring the cell under-contended.
        """
        deadline = time.monotonic() + timeout_s
        seen = 0
        while time.monotonic() < deadline:
            result = self._run(["podman", "top", name, "args"], timeout=10)
            if result.returncode != 0:
                raise RuntimeError(
                    f"stress-ng antagonist '{name}' vanished before verification: "
                    f"{self._tail_logs(name)}"
                )
            seen = sum(1 for l in result.stdout.splitlines() if "stress-ng" in l)
            if seen >= 1 + expected_workers:
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"stress-ng antagonist '{name}' shows {seen} processes after "
            f"{timeout_s:.0f}s, expected >= {1 + expected_workers} (1 supervisor + "
            f"{expected_workers} workers). Refusing to score an under-contended "
            f"cell as contended. {self._tail_logs(name)}"
        )

    def _verify_fence_cpus(self, name: str, cpus: float) -> None:
        result = self._run(
            ["podman", "inspect", "--format", "{{.HostConfig.NanoCpus}}", name],
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Cannot inspect antagonist '{name}' fence: {result.stderr[:200]}")
        try:
            nano = int(result.stdout.strip())
        except ValueError:
            raise RuntimeError(
                f"Unparseable NanoCpus for antagonist '{name}': {result.stdout!r}"
            )
        want = int(cpus * 1_000_000_000)
        if abs(nano - want) > 1_000_000:  # 0.001-core tolerance
            raise RuntimeError(
                f"Antagonist fence not applied: NanoCpus={nano}, wanted {want} "
                f"({cpus:g} cores). An unfenced antagonist is unreproducible."
            )

    def _verify_fence_shares(self, name: str, cpu_shares: int) -> None:
        """Read back the antagonist's CPU weight so a silently-dropped
        ``--cpu-shares`` cannot leave the SLAM uncontended in weight mode."""
        result = self._run(
            ["podman", "inspect", "--format", "{{.HostConfig.CpuShares}}", name],
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Cannot inspect antagonist '{name}' cpu_shares: {result.stderr[:200]}"
            )
        try:
            got = int(result.stdout.strip())
        except ValueError:
            raise RuntimeError(
                f"Unparseable CpuShares for antagonist '{name}': {result.stdout!r}"
            )
        if got != cpu_shares:
            raise RuntimeError(
                f"Antagonist weight not applied: CpuShares={got}, wanted {cpu_shares}. "
                "Without the weight, the SLAM keeps its fair share and the cell is "
                "uncontended."
            )

    # ------------------------------------------------------------------
    # In-container CPU burn (flat competition inside the SLAM's cgroup)
    # ------------------------------------------------------------------

    def _spawn_in_container(self, spec: LoadInContainer) -> None:
        """Exec the framework's static stress-ng INTO the running SLAM container.

        The binary reaches the container via the launch-time bind mount
        (orchestrator.load_launch_config -> base launch extras), so this works
        against any image with a POSIX sh -- no image changes. Workers land in
        the SLAM's own cgroup, competing flat/thread-for-thread at equal
        priority. A sentinel-watching wrapper gives sh-only teardown: touch
        the sentinel and the watcher kills stress-ng.
        """
        target = self._target_container
        if not target:
            raise RuntimeError("In-container load requires the SLAM container name.")

        probe = self._run(
            ["podman", "exec", target, "test", "-x", STRESS_NG_CONTAINER_PATH],
            timeout=15,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                f"Static stress-ng not found at {STRESS_NG_CONTAINER_PATH} inside "
                f"'{target}'. Either the binary is missing on the host (run "
                "deps/stress-ng/extract_static.sh) or the wrapper did not merge "
                "runtime-stress launch extras into its podman run command. "
                "Refusing to run an uncontended cell as contended."
            )

        args = []
        total_workers = 0
        if spec.cpu_workers:
            args += ["--cpu", str(spec.cpu_workers), "--cpu-method", CPU_METHOD]
            total_workers += spec.cpu_workers
        if spec.stream_workers:
            args += ["--stream", str(spec.stream_workers)]
            total_workers += spec.stream_workers
        if spec.vm_workers:
            args += ["--vm", str(spec.vm_workers), "--vm-bytes", f"{spec.vm_bytes_mb}M"]
            total_workers += spec.vm_workers

        stop = self._incontainer_stop
        watcher = (
            f"rm -f {stop}; "
            f"{STRESS_NG_CONTAINER_PATH} {' '.join(args)} & w=$!; "
            f"while [ ! -f {stop} ]; do sleep 0.2; done; "
            f"kill $w 2>/dev/null; wait $w 2>/dev/null"
        )
        result = self._run(["podman", "exec", "-d", target, "sh", "-c", watcher], timeout=30)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to exec in-container stress-ng into '{target}': "
                f"{result.stderr.strip()[:300]}"
            )
        self._incontainer_active = True
        self._incontainer_last_spec = spec
        self._verify_incontainer_count(target, total_workers)
        logger.info(
            "  In-container load: stress-ng %s in %s (flat equal-priority)",
            " ".join(args), target,
        )

    def _teardown_in_container(self) -> None:
        if not self._incontainer_active:
            return
        target = self._target_container
        if target:
            # touch the sentinel -> the watcher kills stress-ng. Tolerate an
            # already-gone container (workers died with it).
            self._run(["podman", "exec", target, "touch", self._incontainer_stop], timeout=15)
        self._incontainer_active = False

    def _verify_incontainer_count(
        self, target: str, expected_workers: int, timeout_s: float = 5.0
    ) -> None:
        """Poll until the requested worker count is visible (fail loud).

        ``podman top`` reads host-side /proc, so no ps is needed in the image.
        stress-ng lines are unmistakable: nothing else in a SLAM container
        contains 'stress-ng'.
        """
        deadline = time.monotonic() + timeout_s
        seen = 0
        while time.monotonic() < deadline:
            result = self._run(["podman", "top", target, "args"], timeout=10)
            if result.returncode == 0:
                seen = sum(1 for l in result.stdout.splitlines() if "stress-ng" in l)
                if seen >= 1 + expected_workers:  # supervisor + workers
                    return
            time.sleep(0.2)
        raise RuntimeError(
            f"In-container stress-ng shows {seen} processes after {timeout_s:.0f}s, "
            f"expected >= {1 + expected_workers} (supervisor + {expected_workers} "
            "workers). Refusing to score an under-contended cell as contended."
        )

    # ------------------------------------------------------------------
    # GPU antagonist
    # ------------------------------------------------------------------

    def _spawn_gpu_antagonist(self) -> None:
        script = Path(__file__).resolve().parent / "gpu_antagonist.py"
        if not script.exists():
            raise RuntimeError(f"gpu_antagonist.py not found at {script}")
        self._gpu_dir = Path(tempfile.mkdtemp(prefix="sal-load-"))
        name = f"sal-load-gpu-{self._session_id}"
        cmd = [
            "podman", "run", "-d", "--rm",
            "--name", name,
            "--label", f"{LOAD_LABEL}=1",
            "--label", f"{LOAD_LABEL}.session={self._session_id}",
            "--cpus", GPU_FENCE_CPUS,
            "--memory", GPU_FENCE_MEMORY,
            "--device", "nvidia.com/gpu=all",
            "-v", f"{script}:{_CONTAINER_SCRIPT}:ro",
            "-v", f"{self._gpu_dir}:{_CONTAINER_LOAD_DIR}:rw",
            self._gpu_image,
            "python3", _CONTAINER_SCRIPT,
            "--control", f"{_CONTAINER_LOAD_DIR}/control.json",
            "--status", f"{_CONTAINER_LOAD_DIR}/status.json",
        ]
        result = self._run(cmd, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to spawn GPU antagonist: {result.stderr.strip()[:300]}"
            )
        self._gpu_container = name

        t0 = time.monotonic()
        deadline = t0 + self._gpu_ready_timeout_s
        status_path = self._gpu_dir / "status.json"
        while time.monotonic() < deadline:
            status = self._read_json(status_path)
            if status is not None:
                if status.get("state") == "ready":
                    self._gpu_ready_elapsed_s = round(time.monotonic() - t0, 2)
                    self._record_digest(self._gpu_image)
                    logger.info(
                        "  GPU antagonist ready in %.1fs (%s)",
                        self._gpu_ready_elapsed_s,
                        name,
                    )
                    return
                if status.get("state") == "error":
                    raise RuntimeError(
                        f"GPU antagonist failed during init: {status.get('message')} "
                        f"{self._tail_logs(name)}"
                    )
            if not self._is_running(name):
                raise RuntimeError(
                    f"GPU antagonist container died during init. {self._tail_logs(name)}"
                )
            time.sleep(0.5)
        raise RuntimeError(
            f"GPU antagonist not ready within {self._gpu_ready_timeout_s:.0f}s. "
            f"{self._tail_logs(name)}"
        )

    def _reconfigure_gpu(self, spec: LoadGpuAntagonist) -> None:
        assert self._gpu_dir is not None
        self._gpu_epoch += 1
        payload = {
            "epoch": self._gpu_epoch,
            "vram_mb": spec.vram_mb,
            "matmul_n": spec.matmul_n,
            "duty_cycle": spec.duty_cycle,
        }
        self._write_json_atomic(self._gpu_dir / "control.json", payload)
        self._gpu_spec = spec

        deadline = time.monotonic() + self._gpu_ack_timeout_s
        status_path = self._gpu_dir / "status.json"
        while time.monotonic() < deadline:
            status = self._read_json(status_path)
            if status is not None:
                if status.get("state") == "error":
                    raise RuntimeError(
                        f"GPU antagonist errored applying epoch {self._gpu_epoch}: "
                        f"{status.get('message')}"
                    )
                if status.get("epoch") == self._gpu_epoch:
                    return
            if not self._is_running(self._gpu_container):
                raise RuntimeError(
                    "GPU antagonist container died while applying "
                    f"epoch {self._gpu_epoch}. {self._tail_logs(self._gpu_container)}"
                )
            time.sleep(0.2)
        raise RuntimeError(
            f"GPU antagonist did not acknowledge epoch {self._gpu_epoch} within "
            f"{self._gpu_ack_timeout_s:.0f}s — refusing to score an uncontended "
            "cell as GPU-contended."
        )

    def _idle_gpu(self) -> None:
        if self._gpu_container is None or self._gpu_dir is None:
            return
        self._reconfigure_gpu(LoadGpuAntagonist(vram_mb=0, matmul_n=0, duty_cycle=0.0))

    def _teardown_gpu(self) -> None:
        if self._gpu_container is None:
            return
        self._rm_container(self._gpu_container)
        self._gpu_container = None

    # ------------------------------------------------------------------
    # podman helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _run(cmd: List[str], timeout: float):
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def _require_image(self, image: str, hint: str) -> None:
        result = self._run(["podman", "image", "exists", image], timeout=15)
        if result.returncode != 0:
            raise RuntimeError(
                f"Antagonist image '{image}' not present. {hint}. "
                "Refusing to run an uncontended cell as contended."
            )

    def _record_digest(self, image: Optional[str]) -> None:
        if not image or image in self._image_digests:
            return
        result = self._run(
            ["podman", "image", "inspect", "--format", "{{.Digest}}", image],
            timeout=10,
        )
        if result.returncode == 0:
            self._image_digests[image] = result.stdout.strip()

    def _await_target_exists(self, name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self._run(["podman", "container", "exists", name], timeout=10)
            if result.returncode == 0:
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"Target container '{name}' did not appear within {timeout_s:.0f}s; "
            "cannot attach load antagonists to a run that never started."
        )

    def _is_running(self, name: Optional[str]) -> bool:
        if not name:
            return False
        result = self._run(
            ["podman", "inspect", "--format", "{{.State.Running}}", name], timeout=10
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _await_running(self, name: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._is_running(name):
                return
            time.sleep(0.2)
        raise RuntimeError(
            f"Antagonist '{name}' did not reach Running within {timeout_s:.0f}s. "
            f"{self._tail_logs(name)}"
        )

    def _rm_container(self, name: str) -> None:
        if self._target_container and name == self._target_container:
            raise RuntimeError(
                f"BUG: attempted to tear down the SLAM container '{name}' as an antagonist"
            )
        result = self._run(["podman", "rm", "-f", name], timeout=20)
        if result.returncode != 0 and "no such container" not in result.stderr.lower():
            logger.warning("  Antagonist teardown of %s: %s", name, result.stderr.strip()[:200])

    def _tail_logs(self, name: Optional[str]) -> str:
        if not name:
            return ""
        result = self._run(["podman", "logs", "--tail", "20", name], timeout=10)
        text = (result.stdout or "") + (result.stderr or "")
        return f"Last logs: {text.strip()[:400]}" if text.strip() else ""

    @staticmethod
    def _read_json(path: Path):
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        tmp.replace(path)
