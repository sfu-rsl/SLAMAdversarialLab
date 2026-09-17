"""Runtime-stress runtime models and compiler helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..config.schema import (
    CpuControlConfig,
    GpuControlConfig,
    IoControlConfig,
    LoadControlConfig,
    MemoryControlConfig,
    RealtimeDeadlineConfig,
    RuntimeStressConfig,
    RuntimeStressControlsConfig,
    RuntimeStressScenarioConfig,
    StressorPresetConfig,
)


@dataclass(frozen=True)
class CpuControl:
    """Normalized CPU control state for one phase."""

    max_cores: Optional[float] = None


@dataclass(frozen=True)
class MemoryControl:
    """Normalized memory control state for one phase."""

    max_mb: Optional[int] = None


@dataclass(frozen=True)
class GpuControl:
    """Normalized GPU control state for a HAMi-backed scenario.

    HAMi reads its env vars once at process init, so these values
    cannot meaningfully change between phases. Scenario-level
    validation enforces that any phase declaring gpu controls agrees.
    """

    vram_limit_mb: Optional[int] = None
    sm_limit_percent: Optional[int] = None


@dataclass(frozen=True)
class IoControl:
    """Normalized block-IO control state for one phase."""

    read_bps: Optional[int] = None
    write_bps: Optional[int] = None
    read_iops: Optional[int] = None
    write_iops: Optional[int] = None


@dataclass(frozen=True)
class LoadGpuAntagonist:
    """Normalized GPU load-antagonist spec (a contending workload, not a cap)."""

    vram_mb: int = 0
    matmul_n: int = 0
    duty_cycle: float = 0.0
    image: Optional[str] = None


@dataclass(frozen=True)
class LoadFence:
    """DEPRECATED (sibling stress-ng calibration). Prefer LoadInContainer.

    Calibration applied to the antagonist container itself.

    ``cpus`` = quota (antagonist capped, SLAM protected by fair share).
    ``cpu_shares`` = weight (antagonist uncapped but high-priority, collapsing
    the SLAM's proportional fair-share slice). At least one is set.
    """

    cpus: Optional[float] = None
    memory_mb: Optional[int] = None
    cpu_shares: Optional[int] = None


@dataclass(frozen=True)
class LoadInContainer:
    """In-container stress-ng workers (run inside the SLAM's own cgroup)."""

    cpu_workers: int = 0
    stream_workers: int = 0
    vm_workers: int = 0
    vm_bytes_mb: int = 0


@dataclass(frozen=True)
class LoadSegmentation:
    """Normalized SAM 3 co-tenant state for one phase.

    `frames_dir` is already RESOLVED here: `source: dataset` is expanded against
    the experiment's dataset block at compile time, so the controller never has
    to know that the reference existed. Keeping the resolution here means an
    unresolvable reference fails while the config is being compiled rather than
    half-way into a run.
    """

    frames_dir: str
    prompt: str
    max_frames: int
    conda_env: str


@dataclass(frozen=True)
class LoadControl:
    """Normalized load-antagonist state for one phase.

    ``cpu_workers``/``stream_workers``/``vm_workers``/``vm_bytes_mb`` + ``fence``
    drive the DEPRECATED sibling stress-ng antagonist; ``in_container`` is the
    supported CPU-side path and ``gpu`` the (supported) GPU antagonist.
    """

    # deprecated sibling stress-ng knobs
    cpu_workers: int = 0
    stream_workers: int = 0
    vm_workers: int = 0
    vm_bytes_mb: int = 0
    fence: Optional[LoadFence] = None
    # supported
    gpu: Optional[LoadGpuAntagonist] = None
    in_container: Optional[LoadInContainer] = None
    # SAM 3 co-tenant: a REAL workload rather than a synthetic one. Saturating
    # and binary -- one instance holds 88-100% SM, so there is no dose to carry.
    segmentation: Optional[LoadSegmentation] = None


@dataclass(frozen=True)
class RuntimeStressControls:
    """Normalized controls for one runtime-stress phase."""

    cpu: Optional[CpuControl] = None
    memory: Optional[MemoryControl] = None
    gpu: Optional[GpuControl] = None
    io: Optional[IoControl] = None
    load: Optional[LoadControl] = None


@dataclass(frozen=True)
class RuntimeStressPhase:
    """Compiled runtime-stress phase.

    Exactly one of ``duration_s`` (wall-clock anchored) or ``until_frame``
    (frame-anchored, the sampled-stream index at which the phase ends) is
    set; the other is None.
    """

    name: str
    duration_s: Optional[float] = None
    until_frame: Optional[int] = None
    controls: RuntimeStressControls = field(default_factory=RuntimeStressControls)


@dataclass(frozen=True)
class RealtimeDeadline:
    """Compiled real-time frame-delivery deadline for a scenario."""

    target_fps: float
    warmup_frames: int = 0
    queue_size: int = 1
    drop_policy: str = "drop_oldest"


@dataclass(frozen=True)
class RuntimeStressRequest:
    """Compiled runtime-stress scenario attached to one SLAM run."""

    scenario_name: str
    telemetry_sample_period_ms: int
    phases: List[RuntimeStressPhase]
    container_runtime: str = "docker"
    gpu: Optional[GpuControl] = None
    realtime: Optional[RealtimeDeadline] = None

    @property
    def frame_anchored(self) -> bool:
        """True when phase boundaries are anchored to processed frames.

        A frame-anchored request advances phases by the SLAM's live frame
        count (reported by the deadline harness) instead of wall-clock
        elapsed time. Schema validation guarantees a scenario is all-time
        or all-frame, so checking the first phase suffices.
        """
        return bool(self.phases) and self.phases[0].until_frame is not None


def _compile_cpu(cfg: Optional[CpuControlConfig]) -> Optional[CpuControl]:
    if cfg is None or cfg.max_cores is None:
        return None
    return CpuControl(max_cores=float(cfg.max_cores))


def _compile_memory(cfg: Optional[MemoryControlConfig]) -> Optional[MemoryControl]:
    if cfg is None or cfg.max_mb is None:
        return None
    return MemoryControl(max_mb=int(cfg.max_mb))


def _compile_gpu(cfg: Optional[GpuControlConfig]) -> Optional[GpuControl]:
    if cfg is None:
        return None
    if cfg.vram_limit_mb is None and cfg.sm_limit_percent is None:
        return None
    return GpuControl(
        vram_limit_mb=int(cfg.vram_limit_mb) if cfg.vram_limit_mb is not None else None,
        sm_limit_percent=int(cfg.sm_limit_percent) if cfg.sm_limit_percent is not None else None,
    )


def _compile_io(cfg: Optional[IoControlConfig]) -> Optional[IoControl]:
    if cfg is None:
        return None
    if (
        cfg.read_bps is None
        and cfg.write_bps is None
        and cfg.read_iops is None
        and cfg.write_iops is None
    ):
        return None
    return IoControl(
        read_bps=int(cfg.read_bps) if cfg.read_bps is not None else None,
        write_bps=int(cfg.write_bps) if cfg.write_bps is not None else None,
        read_iops=int(cfg.read_iops) if cfg.read_iops is not None else None,
        write_iops=int(cfg.write_iops) if cfg.write_iops is not None else None,
    )


def _compile_load(cfg: Optional[LoadControlConfig]) -> Optional[LoadControl]:
    if cfg is None:
        return None
    gpu = None
    if cfg.gpu is not None:
        gpu = LoadGpuAntagonist(
            vram_mb=int(cfg.gpu.vram_mb) if cfg.gpu.vram_mb is not None else 0,
            matmul_n=int(cfg.gpu.matmul_n) if cfg.gpu.matmul_n is not None else 0,
            duty_cycle=float(cfg.gpu.duty_cycle) if cfg.gpu.duty_cycle is not None else 0.0,
            image=cfg.gpu.image,
        )
    fence = None
    if cfg.fence is not None:
        fence = LoadFence(
            cpus=float(cfg.fence.cpus) if cfg.fence.cpus is not None else None,
            memory_mb=int(cfg.fence.memory_mb) if cfg.fence.memory_mb is not None else None,
            cpu_shares=int(cfg.fence.cpu_shares) if cfg.fence.cpu_shares is not None else None,
        )
    in_container = None
    if cfg.in_container is not None and any(
        (cfg.in_container.cpu_workers, cfg.in_container.stream_workers,
         cfg.in_container.vm_workers)
    ):
        ic = cfg.in_container
        in_container = LoadInContainer(
            cpu_workers=int(ic.cpu_workers) if ic.cpu_workers is not None else 0,
            stream_workers=int(ic.stream_workers) if ic.stream_workers is not None else 0,
            vm_workers=int(ic.vm_workers) if ic.vm_workers is not None else 0,
            vm_bytes_mb=int(ic.vm_bytes_mb) if ic.vm_bytes_mb is not None else 0,
        )
    segmentation = None
    if cfg.segmentation is not None:
        seg = cfg.segmentation
        if not seg.frames_dir:
            # source: dataset is expanded at parse time (parser
            # _resolve_segmentation_source). Reaching here without a concrete
            # directory means that resolution did not happen, and running a cell
            # with no antagonist is exactly the silent-uncontended failure the
            # harness exists to prevent.
            raise ValueError(
                "segmentation load has no frames_dir after config resolution; "
                "refusing to compile a phase whose co-tenant would not run"
            )
        segmentation = LoadSegmentation(
            frames_dir=str(seg.frames_dir),
            prompt=str(seg.prompt),
            max_frames=int(seg.max_frames) if seg.max_frames else 0,
            conda_env=str(seg.conda_env),
        )
    load = LoadControl(
        cpu_workers=int(cfg.cpu_workers) if cfg.cpu_workers is not None else 0,
        stream_workers=int(cfg.stream_workers) if cfg.stream_workers is not None else 0,
        vm_workers=int(cfg.vm_workers) if cfg.vm_workers is not None else 0,
        vm_bytes_mb=int(cfg.vm_bytes_mb) if cfg.vm_bytes_mb is not None else 0,
        gpu=gpu,
        fence=fence,
        in_container=in_container,
        segmentation=segmentation,
    )
    if (
        load.cpu_workers == 0
        and load.stream_workers == 0
        and load.vm_workers == 0
        and load.gpu is None
        and load.in_container is None
        and load.segmentation is None
    ):
        return None
    return load


def _compile_controls_block(
    cpu: Optional[CpuControlConfig],
    memory: Optional[MemoryControlConfig],
    gpu: Optional[GpuControlConfig],
    io: Optional[IoControlConfig],
    load: Optional[LoadControlConfig] = None,
) -> RuntimeStressControls:
    """Compile an axis config block into runtime ``RuntimeStressControls``."""
    return RuntimeStressControls(
        cpu=_compile_cpu(cpu),
        memory=_compile_memory(memory),
        gpu=_compile_gpu(gpu),
        io=_compile_io(io),
        load=_compile_load(load),
    )


def _min_optional(a, b):
    """Return the tighter (smaller) non-None value, or whichever is non-None, or None."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a <= b else b


def _merge_controls_tighter_wins(
    *candidates: RuntimeStressControls,
) -> RuntimeStressControls:
    """Merge multiple controls into one using tighter-wins per axis-field.

    Tighter-wins: for every numeric field, the result takes the smallest
    non-None value across candidates. None is treated as "unset" and
    skipped. If every candidate has the field unset, the result has it
    unset too.

    ``load`` is single-source: min-wins is meaningless for an antagonist
    spec (a larger fence is LESS stress), so exactly one candidate may
    carry a load block and it passes through unchanged. Two load-carrying
    candidates raise. (Presets cannot carry load in v1, so in practice
    the single source is always the phase's inline controls.)
    """
    merged_cpu_max_cores: Optional[float] = None
    merged_memory_max_mb: Optional[int] = None
    merged_vram_limit_mb: Optional[int] = None
    merged_sm_limit_percent: Optional[int] = None
    merged_read_bps: Optional[int] = None
    merged_write_bps: Optional[int] = None
    merged_read_iops: Optional[int] = None
    merged_write_iops: Optional[int] = None

    merged_load: Optional[LoadControl] = None

    for ctrl in candidates:
        if ctrl.load is not None:
            if merged_load is not None:
                raise ValueError(
                    "multiple load blocks cannot be merged (load is single-source in v1)"
                )
            merged_load = ctrl.load
        if ctrl.cpu is not None:
            merged_cpu_max_cores = _min_optional(merged_cpu_max_cores, ctrl.cpu.max_cores)
        if ctrl.memory is not None:
            merged_memory_max_mb = _min_optional(merged_memory_max_mb, ctrl.memory.max_mb)
        if ctrl.gpu is not None:
            merged_vram_limit_mb = _min_optional(merged_vram_limit_mb, ctrl.gpu.vram_limit_mb)
            merged_sm_limit_percent = _min_optional(merged_sm_limit_percent, ctrl.gpu.sm_limit_percent)
        if ctrl.io is not None:
            merged_read_bps = _min_optional(merged_read_bps, ctrl.io.read_bps)
            merged_write_bps = _min_optional(merged_write_bps, ctrl.io.write_bps)
            merged_read_iops = _min_optional(merged_read_iops, ctrl.io.read_iops)
            merged_write_iops = _min_optional(merged_write_iops, ctrl.io.write_iops)

    cpu = CpuControl(max_cores=merged_cpu_max_cores) if merged_cpu_max_cores is not None else None
    memory = MemoryControl(max_mb=merged_memory_max_mb) if merged_memory_max_mb is not None else None

    gpu: Optional[GpuControl] = None
    if merged_vram_limit_mb is not None or merged_sm_limit_percent is not None:
        gpu = GpuControl(
            vram_limit_mb=merged_vram_limit_mb,
            sm_limit_percent=merged_sm_limit_percent,
        )

    io: Optional[IoControl] = None
    if (
        merged_read_bps is not None
        or merged_write_bps is not None
        or merged_read_iops is not None
        or merged_write_iops is not None
    ):
        io = IoControl(
            read_bps=merged_read_bps,
            write_bps=merged_write_bps,
            read_iops=merged_read_iops,
            write_iops=merged_write_iops,
        )

    return RuntimeStressControls(cpu=cpu, memory=memory, gpu=gpu, io=io, load=merged_load)


def _compile_preset(preset: StressorPresetConfig) -> RuntimeStressControls:
    return _compile_controls_block(preset.cpu, preset.memory, preset.gpu, preset.io, None)


def _compile_phase_controls(
    phase_controls: RuntimeStressControlsConfig,
) -> RuntimeStressControls:
    return _compile_controls_block(
        phase_controls.cpu,
        phase_controls.memory,
        phase_controls.gpu,
        phase_controls.io,
        phase_controls.load,
    )


def compile_runtime_stress_request(
    runtime_stress: RuntimeStressConfig,
    scenario: RuntimeStressScenarioConfig,
) -> RuntimeStressRequest:
    """Compile validated config-schema objects into runtime models.

    For each phase, the effective ``RuntimeStressControls`` is built by
    merging the resolved stressor presets (in list order) with the
    inline ``controls`` block. The merge rule is tighter-wins: smaller
    is more constrained. Phase ``stressors`` references must already
    have been validated against ``runtime_stress.stressors`` at schema
    time.
    """
    compiled_phases: List[RuntimeStressPhase] = []
    for phase in scenario.phases:
        candidates: List[RuntimeStressControls] = []
        for ref in phase.stressors:
            preset = runtime_stress.stressors.get(ref)
            if preset is None:
                continue
            candidates.append(_compile_preset(preset))
        candidates.append(_compile_phase_controls(phase.controls))

        merged = _merge_controls_tighter_wins(*candidates)

        compiled_phases.append(
            RuntimeStressPhase(
                name=phase.name.strip(),
                duration_s=float(phase.duration_s) if phase.duration_s is not None else None,
                until_frame=int(phase.until_frame) if phase.until_frame is not None else None,
                controls=merged,
            )
        )

    resolved_gpu: Optional[GpuControl] = None
    for compiled_phase in compiled_phases:
        if compiled_phase.controls.gpu is not None:
            resolved_gpu = compiled_phase.controls.gpu
            break

    realtime: Optional[RealtimeDeadline] = None
    if scenario.realtime is not None:
        realtime = RealtimeDeadline(
            target_fps=float(scenario.realtime.target_fps),
            warmup_frames=int(scenario.realtime.warmup_frames),
            queue_size=int(scenario.realtime.queue_size),
            drop_policy=scenario.realtime.drop_policy,
        )

    return RuntimeStressRequest(
        scenario_name=scenario.name.strip(),
        telemetry_sample_period_ms=runtime_stress.telemetry.sample_period_ms,
        phases=compiled_phases,
        container_runtime=runtime_stress.container_runtime,
        gpu=resolved_gpu,
        realtime=realtime,
    )
