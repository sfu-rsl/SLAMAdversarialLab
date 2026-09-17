"""YAML configuration parser for SLAMAdversarialLab."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from ..utils import get_logger
from .schema import (
    ExperimentConfig,
    DatasetConfig,
    PerturbationConfig,
    OutputConfig,
    RealtimeDeadlineConfig,
    RobustnessBoundaryConfig,
    RuntimeStressConfig,
    RuntimeStressTelemetryConfig,
    RuntimeStressScenarioConfig,
    RuntimeStressPhaseConfig,
    RuntimeStressControlsConfig,
    CpuControlConfig,
    GpuControlConfig,
    IoControlConfig,
    LoadControlConfig,
    LoadFenceConfig,
    LoadGpuAntagonistConfig,
    LoadInContainerConfig,
    LoadSegmentationConfig,
    MemoryControlConfig,
    StressorPresetConfig,
)

logger = get_logger(__name__)


def expand_env_vars(value: Any) -> Any:
    """
    Recursively expand environment variables in configuration values.

    Args:
        value: Configuration value (string, list, dict, or other)

    Returns:
        Value with environment variables expanded
    """
    if isinstance(value, str):
        # Pattern matches ${VAR} or $VAR
        pattern = re.compile(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)')

        def replacer(match):
            var_name = match.group(1) or match.group(2)
            var_value = os.environ.get(var_name, "")
            if not var_value:
                logger.warning(f"Environment variable '{var_name}' not found")
            return var_value

        return pattern.sub(replacer, value)
    elif isinstance(value, dict):
        return {k: expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    else:
        return value


def parse_experiment(config_dict: Dict[str, Any]) -> ExperimentConfig:
    """
    Parse experiment configuration section.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        ExperimentConfig instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    experiment_data = config_dict.get("experiment", {})

    if not experiment_data.get("name"):
        raise ValueError("experiment.name is required in configuration")

    experiment = ExperimentConfig(
        name=experiment_data["name"],
        description=experiment_data.get("description", ""),
        version=experiment_data.get("version", "1.0.0"),
        seed=experiment_data.get("seed")
    )

    experiment.validate()
    return experiment


def parse_dataset(config_dict: Dict[str, Any]) -> DatasetConfig:
    """
    Parse dataset configuration section.

    Supports two modes:
    1. Explicit path: dataset.path is provided directly
    2. Sequence-based: dataset.sequence is provided, path is auto-resolved
       - For TUM: auto-downloads if missing
       - For KITTI: looks in standard location

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        DatasetConfig instance

    Raises:
        ValueError: If required fields are missing or invalid
    """
    dataset_data = config_dict.get("dataset", {})

    if not dataset_data:
        raise ValueError("dataset section is required in configuration")

    if not dataset_data.get("type"):
        raise ValueError("dataset.type is required")

    if "sequences" in dataset_data:
        raise ValueError(
            "dataset.sequences is no longer supported. "
            "Use singular 'dataset.sequence' instead."
        )

    # Either path or sequence must be provided (except for mock datasets)
    has_path = bool(dataset_data.get("path"))
    has_sequence = bool(dataset_data.get("sequence"))

    if not has_path and not has_sequence and dataset_data["type"] != "mock":
        raise ValueError(
            "dataset requires either 'path' or 'sequence'. "
            "Use 'sequence' for auto-resolution (e.g., sequence: freiburg1_desk for TUM)"
        )

    # Expand environment variables in path if provided
    path = dataset_data.get("path")
    if path:
        path = expand_env_vars(path)

    dataset = DatasetConfig(
        type=dataset_data["type"],
        path=path,
        sequence=dataset_data.get("sequence"),
        max_frames=dataset_data.get("max_frames"),
        load_stereo=dataset_data.get("load_stereo", False),
        skip_depth=dataset_data.get("skip_depth", False)
    )

    dataset.validate()
    return dataset


def parse_perturbations(config_dict: Dict[str, Any]) -> List[PerturbationConfig]:
    """
    Parse perturbations configuration section.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        List of PerturbationConfig instances

    Raises:
        ValueError: If perturbation configuration is invalid
    """
    perturbations_data = config_dict.get("perturbations", [])
    perturbations = []

    for idx, pert_data in enumerate(perturbations_data):
        if not pert_data.get("name"):
            raise ValueError(f"perturbations[{idx}].name is required")

        if not pert_data.get("type"):
            raise ValueError(f"perturbations[{idx}].type is required")

        perturbation = PerturbationConfig(
            name=pert_data["name"],
            type=pert_data["type"],
            enabled=pert_data.get("enabled", True),
            parameters=pert_data.get("parameters", {})
        )

        perturbation.validate()
        perturbations.append(perturbation)

    # If no perturbations specified, add a default "none" perturbation
    if not perturbations:
        logger.info("No perturbations specified, using default 'none' perturbation")
        perturbations.append(PerturbationConfig(
            name="baseline",
            type="none",
            enabled=True,
            parameters={}
        ))

    return perturbations


def parse_profiling(config_dict: Dict[str, Any]) -> Optional['ProfilingConfig']:
    """
    Parse profiling configuration section.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        ProfilingConfig object or None if not present
    """
    if 'profiling' not in config_dict:
        return None

    from .schema import ProfilingConfig

    profiling_data = config_dict['profiling']
    return ProfilingConfig(
        enabled=profiling_data.get('enabled', False),
        verbose=profiling_data.get('verbose', False),
        save_report=profiling_data.get('save_report', True),
        report_format=profiling_data.get('report_format', 'json'),
        report_path=profiling_data.get('report_path', None)
    )


def parse_slam(config_dict: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse SLAM configuration section.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        SLAM config dict or None if not present
    """
    if 'slam' not in config_dict:
        return None

    slam_data = config_dict['slam']

    from types import SimpleNamespace
    return SimpleNamespace(
        algorithms=slam_data.get('algorithms', []),
        metrics=slam_data.get('metrics', ['ate', 'rpe']),
        visualize=slam_data.get('visualize', False)
    )


def parse_robustness_boundary(config_dict: Dict[str, Any]) -> Optional[RobustnessBoundaryConfig]:
    """
    Parse robustness-boundary configuration section.

    Supports preferred `lower_bound`/`upper_bound` keys and legacy `low`/`high`.

    Args:
        config_dict: Full configuration dictionary

    Returns:
        RobustnessBoundaryConfig object or None if not present
    """
    if 'robustness_boundary' not in config_dict:
        return None

    rb_data = config_dict['robustness_boundary']
    if rb_data is None:
        return None

    if not isinstance(rb_data, dict):
        raise ValueError("robustness_boundary must be a dictionary/object")

    # Backward-compatibility: accept low/high if lower_bound/upper_bound are absent.
    lower_bound = rb_data.get('lower_bound', rb_data.get('low'))
    upper_bound = rb_data.get('upper_bound', rb_data.get('high'))
    if ('low' in rb_data or 'high' in rb_data) and (
        'lower_bound' not in rb_data or 'upper_bound' not in rb_data
    ):
        logger.warning(
            "robustness_boundary.low/high is deprecated. "
            "Use lower_bound/upper_bound instead."
        )

    # Backward-compatibility: accept ape_rmse_fail when ate_rmse_fail is absent.
    ate_rmse_fail = rb_data.get('ate_rmse_fail', rb_data.get('ape_rmse_fail', 1.5))
    if 'ape_rmse_fail' in rb_data and 'ate_rmse_fail' not in rb_data:
        logger.warning(
            "robustness_boundary.ape_rmse_fail is deprecated. "
            "Use ate_rmse_fail instead."
        )

    rb = RobustnessBoundaryConfig(
        enabled=rb_data.get('enabled', False),
        name=rb_data.get('name', ''),
        target_perturbation=rb_data.get('target_perturbation', ''),
        module=rb_data.get('module', ''),
        parameter=rb_data.get('parameter', ''),
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        tolerance=rb_data.get('tolerance', 0.05),
        max_iters=rb_data.get('max_iters', 8),
        ate_rmse_fail=ate_rmse_fail,
        fail_on_tracking_failure=rb_data.get('fail_on_tracking_failure', True),
    )
    rb.validate()
    return rb


def _parse_axis_blocks(
    controls_data: Dict[str, Any], context: str
) -> Tuple[
    Optional[CpuControlConfig],
    Optional[MemoryControlConfig],
    Optional[GpuControlConfig],
    Optional[IoControlConfig],
    Optional[LoadControlConfig],
]:
    """Parse the axis blocks (cpu/memory/gpu/io/load) shared by phase controls
    and stressor presets.

    ``context`` is a YAML-path prefix used in error messages
    (e.g. ``runtime_stress.scenarios[0].phases[1].controls`` or
    ``runtime_stress.stressors['edge_device']``).
    """
    cpu_data = controls_data.get("cpu")
    if cpu_data is not None and not isinstance(cpu_data, dict):
        raise ValueError(f"{context}.cpu must be a dictionary/object")
    cpu_cfg = (
        CpuControlConfig(max_cores=cpu_data.get("max_cores"))
        if cpu_data is not None
        else None
    )

    memory_data = controls_data.get("memory")
    if memory_data is not None and not isinstance(memory_data, dict):
        raise ValueError(f"{context}.memory must be a dictionary/object")
    memory_cfg = (
        MemoryControlConfig(max_mb=memory_data.get("max_mb"))
        if memory_data is not None
        else None
    )

    gpu_data = controls_data.get("gpu")
    if gpu_data is not None and not isinstance(gpu_data, dict):
        raise ValueError(f"{context}.gpu must be a dictionary/object")
    gpu_cfg = (
        GpuControlConfig(
            vram_limit_mb=gpu_data.get("vram_limit_mb"),
            sm_limit_percent=gpu_data.get("sm_limit_percent"),
        )
        if gpu_data is not None
        else None
    )

    io_data = controls_data.get("io")
    if io_data is not None and not isinstance(io_data, dict):
        raise ValueError(f"{context}.io must be a dictionary/object")
    io_cfg = (
        IoControlConfig(
            read_bps=io_data.get("read_bps"),
            write_bps=io_data.get("write_bps"),
            read_iops=io_data.get("read_iops"),
            write_iops=io_data.get("write_iops"),
        )
        if io_data is not None
        else None
    )

    load_data = controls_data.get("load")
    if load_data is not None and not isinstance(load_data, dict):
        raise ValueError(f"{context}.load must be a dictionary/object")
    load_cfg: Optional[LoadControlConfig] = None
    if load_data is not None:
        load_gpu_data = load_data.get("gpu")
        if load_gpu_data is not None and not isinstance(load_gpu_data, dict):
            raise ValueError(f"{context}.load.gpu must be a dictionary/object")
        load_fence_data = load_data.get("fence")
        if load_fence_data is not None and not isinstance(load_fence_data, dict):
            raise ValueError(f"{context}.load.fence must be a dictionary/object")
        load_incontainer_data = load_data.get("in_container")
        if load_incontainer_data is not None and not isinstance(load_incontainer_data, dict):
            raise ValueError(f"{context}.load.in_container must be a dictionary/object")
        load_seg_data = load_data.get("segmentation")
        if load_seg_data is not None and not isinstance(load_seg_data, dict):
            raise ValueError(f"{context}.load.segmentation must be a dictionary/object")
        load_cfg = LoadControlConfig(
            cpu_workers=load_data.get("cpu_workers"),
            stream_workers=load_data.get("stream_workers"),
            vm_workers=load_data.get("vm_workers"),
            vm_bytes_mb=load_data.get("vm_bytes_mb"),
            in_container=(
                LoadInContainerConfig(
                    cpu_workers=load_incontainer_data.get("cpu_workers"),
                    stream_workers=load_incontainer_data.get("stream_workers"),
                    vm_workers=load_incontainer_data.get("vm_workers"),
                    vm_bytes_mb=load_incontainer_data.get("vm_bytes_mb"),
                )
                if load_incontainer_data is not None
                else None
            ),
            segmentation=(
                LoadSegmentationConfig(
                    frames_dir=load_seg_data.get("frames_dir"),
                    source=load_seg_data.get("source"),
                    prompt=load_seg_data.get("prompt", "person"),
                    max_frames=load_seg_data.get("max_frames"),
                    conda_env=load_seg_data.get("conda_env", "sam3"),
                )
                if load_seg_data is not None
                else None
            ),
            gpu=(
                LoadGpuAntagonistConfig(
                    vram_mb=load_gpu_data.get("vram_mb"),
                    matmul_n=load_gpu_data.get("matmul_n"),
                    duty_cycle=load_gpu_data.get("duty_cycle"),
                    image=load_gpu_data.get("image"),
                )
                if load_gpu_data is not None
                else None
            ),
            fence=(
                LoadFenceConfig(
                    cpus=load_fence_data.get("cpus"),
                    memory_mb=load_fence_data.get("memory_mb"),
                    cpu_shares=load_fence_data.get("cpu_shares"),
                )
                if load_fence_data is not None
                else None
            ),
        )

    return cpu_cfg, memory_cfg, gpu_cfg, io_cfg, load_cfg


def parse_runtime_stress(config_dict: Dict[str, Any]) -> Optional[RuntimeStressConfig]:
    """Parse runtime-stress configuration section."""
    if "runtime_stress" not in config_dict:
        return None

    rt_data = config_dict["runtime_stress"]
    if rt_data is None:
        return None

    if not isinstance(rt_data, dict):
        raise ValueError("runtime_stress must be a dictionary/object")

    container_runtime = rt_data.get("container_runtime", "docker")
    if not isinstance(container_runtime, str):
        raise ValueError("runtime_stress.container_runtime must be a string")

    telemetry_data = rt_data.get("telemetry", {}) or {}
    if not isinstance(telemetry_data, dict):
        raise ValueError("runtime_stress.telemetry must be a dictionary/object")

    telemetry = RuntimeStressTelemetryConfig(
        sample_period_ms=telemetry_data.get("sample_period_ms", 500),
    )

    raw_stressors = rt_data.get("stressors", {}) or {}
    if not isinstance(raw_stressors, dict):
        raise ValueError("runtime_stress.stressors must be a mapping of name -> preset")

    stressors: Dict[str, StressorPresetConfig] = {}
    for preset_name, preset_data in raw_stressors.items():
        if not isinstance(preset_name, str):
            raise ValueError("runtime_stress.stressors keys must be strings")
        if not isinstance(preset_data, dict):
            raise ValueError(
                f"runtime_stress.stressors['{preset_name}'] must be a dictionary/object"
            )
        cpu_cfg, memory_cfg, gpu_cfg, io_cfg, load_cfg = _parse_axis_blocks(
            preset_data, f"runtime_stress.stressors['{preset_name}']"
        )
        if load_cfg is not None:
            raise ValueError(
                f"runtime_stress.stressors['{preset_name}'].load: load antagonists "
                "are phase-controls-only in v1 (no preset merge semantics defined)"
            )
        stressors[preset_name] = StressorPresetConfig(
            cpu=cpu_cfg, memory=memory_cfg, gpu=gpu_cfg, io=io_cfg
        )

    scenarios: List[RuntimeStressScenarioConfig] = []
    raw_scenarios = rt_data.get("scenarios", [])
    if raw_scenarios is None:
        raw_scenarios = []
    if not isinstance(raw_scenarios, list):
        raise ValueError("runtime_stress.scenarios must be a list")

    for scenario_idx, scenario_data in enumerate(raw_scenarios):
        if not isinstance(scenario_data, dict):
            raise ValueError(
                f"runtime_stress.scenarios[{scenario_idx}] must be a dictionary/object"
            )

        raw_phases = scenario_data.get("phases", [])
        if not isinstance(raw_phases, list):
            raise ValueError(
                f"runtime_stress.scenarios[{scenario_idx}].phases must be a list"
            )

        phases: List[RuntimeStressPhaseConfig] = []
        for phase_idx, phase_data in enumerate(raw_phases):
            if not isinstance(phase_data, dict):
                raise ValueError(
                    f"runtime_stress.scenarios[{scenario_idx}].phases[{phase_idx}] "
                    "must be a dictionary/object"
                )

            controls_data = phase_data.get("controls", {}) or {}
            if not isinstance(controls_data, dict):
                raise ValueError(
                    f"runtime_stress.scenarios[{scenario_idx}].phases[{phase_idx}].controls "
                    "must be a dictionary/object"
                )

            cpu_cfg, memory_cfg, gpu_cfg, io_cfg, load_cfg = _parse_axis_blocks(
                controls_data,
                f"runtime_stress.scenarios[{scenario_idx}].phases[{phase_idx}].controls",
            )

            stressor_refs = phase_data.get("stressors", []) or []
            if not isinstance(stressor_refs, list):
                raise ValueError(
                    f"runtime_stress.scenarios[{scenario_idx}].phases[{phase_idx}].stressors "
                    "must be a list of names"
                )

            phases.append(
                RuntimeStressPhaseConfig(
                    name=phase_data.get("name", ""),
                    duration_s=phase_data.get("duration_s"),
                    until_frame=phase_data.get("until_frame"),
                    controls=RuntimeStressControlsConfig(
                        cpu=cpu_cfg, memory=memory_cfg, gpu=gpu_cfg, io=io_cfg,
                        load=load_cfg,
                    ),
                    stressors=list(stressor_refs),
                )
            )

        realtime_data = scenario_data.get("realtime")
        realtime_cfg: Optional[RealtimeDeadlineConfig] = None
        if realtime_data is not None:
            if not isinstance(realtime_data, dict):
                raise ValueError(
                    f"runtime_stress.scenarios[{scenario_idx}].realtime "
                    "must be a dictionary/object"
                )
            realtime_cfg = RealtimeDeadlineConfig(
                target_fps=realtime_data.get("target_fps"),
                warmup_frames=realtime_data.get("warmup_frames", 0),
                queue_size=realtime_data.get("queue_size", 1),
                drop_policy=realtime_data.get("drop_policy", "drop_oldest"),
            )

        scenarios.append(
            RuntimeStressScenarioConfig(
                name=scenario_data.get("name", ""),
                enabled=scenario_data.get("enabled", True),
                phases=phases,
                realtime=realtime_cfg,
            )
        )

    runtime_stress = RuntimeStressConfig(
        enabled=rt_data.get("enabled", False),
        container_runtime=container_runtime,
        evict_page_cache=rt_data.get("evict_page_cache", False),
        telemetry=telemetry,
        stressors=stressors,
        scenarios=scenarios,
    )
    runtime_stress.validate()
    return runtime_stress


def parse_output(config_dict: Dict[str, Any]) -> OutputConfig:
    """
    Parse output configuration section.

    Args:
        config_dict: Raw configuration dictionary

    Returns:
        OutputConfig instance

    Raises:
        ValueError: If output configuration is invalid
    """
    output_data = config_dict.get("output", {})

    # Expand environment variables in base_dir
    if "base_dir" in output_data:
        output_data["base_dir"] = expand_env_vars(output_data["base_dir"])

    output = OutputConfig(
        base_dir=output_data.get("base_dir", "./results"),
        save_images=output_data.get("save_images", True),
        create_timestamp_dir=output_data.get("create_timestamp_dir", True)
    )

    output.validate()
    return output


def _resolve_segmentation_source(dataset, runtime_stress) -> None:
    """Expand `segmentation.source: dataset` into a concrete frames_dir.

    Done HERE because this is the first point where the dataset block and the
    runtime-stress block both exist. Resolving at parse time means an
    unresolvable reference fails while the config is being read, rather than
    half-way into a campaign when the antagonist tries to open a directory that
    was never there.

    THE REFERENCE IS TO DATA, NOT TO RUNTIME. Exactly four things are copied --
    path, max_frames, and (through the path) the sequence and stereo layout. The
    segmenter does NOT inherit the deadline, the frame pacing, the warmup or the
    phases. "Same dataset" means the same images and the same frame budget, which
    is the realistic part (one camera feeding two consumers), without inventing a
    synchronisation the experiment does not have.
    """
    if runtime_stress is None or dataset is None:
        return
    for scenario in (runtime_stress.scenarios or []):
        for phase in (scenario.phases or []):
            controls = getattr(phase, "controls", None)
            seg = getattr(getattr(controls, "load", None), "segmentation", None)
            if seg is None or seg.source != "dataset":
                continue
            resolved = dataset.path
            if not resolved and dataset.sequence:
                # Sequence-based configs (KITTI uses `sequence: 07` with no
                # path) are resolved by the dataset adapter, not by the config.
                # Reuse THAT resolution rather than reimplementing the layout
                # convention here, so the co-tenant reads the same directory the
                # SLAM will and the two cannot drift apart.
                try:
                    from src.datasets.factory import _registry  # noqa: PLC0415
                    adapter = _registry.get(dataset.type)
                    if adapter is not None:
                        resolved = adapter.resolve_path(dataset)
                except Exception as exc:
                    raise ValueError(
                        f"runtime_stress.controls.load.segmentation uses "
                        f"source: dataset, but sequence "
                        f"{dataset.sequence!r} could not be resolved to a path "
                        f"({exc}). Set an explicit frames_dir instead."
                    ) from exc
            if not resolved:
                raise ValueError(
                    "runtime_stress.controls.load.segmentation uses "
                    "source: dataset, but the dataset block has no resolved "
                    "path. Give the dataset an explicit path, or set an "
                    "explicit frames_dir on the segmentation block."
                )
            seg.frames_dir = resolved
            # Inherit the SLAM's frame budget unless the block overrode it, so
            # `source: dataset` genuinely means the same span of the sequence.
            if seg.max_frames is None:
                seg.max_frames = dataset.max_frames
            seg.source = None  # resolved; downstream sees only a concrete dir


class Config:
    """Complete configuration container."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        dataset: DatasetConfig,
        perturbations: List[PerturbationConfig],
        output: OutputConfig,
        robustness_boundary: Optional[RobustnessBoundaryConfig] = None,
        runtime_stress: Optional[RuntimeStressConfig] = None,
    ):
        """
        Initialize configuration.

        Args:
            experiment: Experiment configuration
            dataset: Dataset configuration
            perturbations: List of perturbation configurations
            output: Output configuration
            robustness_boundary: Optional robustness-boundary configuration
            runtime_stress: Optional runtime-stress configuration
        """
        self.experiment = experiment
        self.dataset = dataset
        self.perturbations = perturbations
        self.output = output
        self.robustness_boundary = robustness_boundary
        self.runtime_stress = runtime_stress
        _resolve_segmentation_source(self.dataset, self.runtime_stress)

    def __repr__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(\n"
            f"  experiment={self.experiment.name},\n"
            f"  dataset={self.dataset.type},\n"
            f"  perturbations=[{', '.join(p.name for p in self.perturbations)}],\n"
            f"  output={self.output.base_dir}\n"
            f")"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {
            "experiment": {
                "name": self.experiment.name,
                "description": self.experiment.description,
                "version": self.experiment.version,
                "seed": self.experiment.seed
            },
            "dataset": {
                "type": self.dataset.type,
                "path": self.dataset.path,
                "sequence": self.dataset.sequence,
                "max_frames": self.dataset.max_frames,
                "skip_depth": self.dataset.skip_depth,
                "load_stereo": self.dataset.load_stereo,
            },
            "perturbations": [
                {
                    "name": p.name,
                    "type": p.type,
                    "enabled": p.enabled,
                    "parameters": p.parameters
                }
                for p in self.perturbations
            ],
            "output": {
                "base_dir": self.output.base_dir,
                "save_images": self.output.save_images,
                "create_timestamp_dir": self.output.create_timestamp_dir
            }
        }

        if self.robustness_boundary is not None:
            config_dict["robustness_boundary"] = {
                "enabled": self.robustness_boundary.enabled,
                "name": self.robustness_boundary.name,
                "target_perturbation": self.robustness_boundary.target_perturbation,
                "module": self.robustness_boundary.module,
                "parameter": self.robustness_boundary.parameter,
                "lower_bound": self.robustness_boundary.lower_bound,
                "upper_bound": self.robustness_boundary.upper_bound,
                "tolerance": self.robustness_boundary.tolerance,
                "max_iters": self.robustness_boundary.max_iters,
                "ate_rmse_fail": self.robustness_boundary.ate_rmse_fail,
                "fail_on_tracking_failure": self.robustness_boundary.fail_on_tracking_failure,
            }

        if self.runtime_stress is not None:
            config_dict["runtime_stress"] = {
                "enabled": self.runtime_stress.enabled,
                "container_runtime": self.runtime_stress.container_runtime,
                "telemetry": {
                    "sample_period_ms": self.runtime_stress.telemetry.sample_period_ms,
                },
                "scenarios": [
                    {
                        "name": scenario.name,
                        "enabled": scenario.enabled,
                        "phases": [
                            {
                                "name": phase.name,
                                "duration_s": phase.duration_s,
                                "until_frame": phase.until_frame,
                                "controls": {
                                    "cpu": (
                                        {"max_cores": phase.controls.cpu.max_cores}
                                        if phase.controls.cpu is not None
                                        else None
                                    ),
                                    "memory": (
                                        {"max_mb": phase.controls.memory.max_mb}
                                        if phase.controls.memory is not None
                                        else None
                                    ),
                                    "gpu": (
                                        {
                                            "vram_limit_mb": phase.controls.gpu.vram_limit_mb,
                                            "sm_limit_percent": phase.controls.gpu.sm_limit_percent,
                                        }
                                        if phase.controls.gpu is not None
                                        else None
                                    ),
                                    "io": (
                                        {
                                            "read_bps": phase.controls.io.read_bps,
                                            "write_bps": phase.controls.io.write_bps,
                                            "read_iops": phase.controls.io.read_iops,
                                            "write_iops": phase.controls.io.write_iops,
                                        }
                                        if phase.controls.io is not None
                                        else None
                                    ),
                                    "load": (
                                        {
                                            "cpu_workers": phase.controls.load.cpu_workers,
                                            "stream_workers": phase.controls.load.stream_workers,
                                            "vm_workers": phase.controls.load.vm_workers,
                                            "vm_bytes_mb": phase.controls.load.vm_bytes_mb,
                                            "gpu": (
                                                {
                                                    "vram_mb": phase.controls.load.gpu.vram_mb,
                                                    "matmul_n": phase.controls.load.gpu.matmul_n,
                                                    "duty_cycle": phase.controls.load.gpu.duty_cycle,
                                                    "image": phase.controls.load.gpu.image,
                                                }
                                                if phase.controls.load.gpu is not None
                                                else None
                                            ),
                                            "fence": (
                                                {
                                                    "cpus": phase.controls.load.fence.cpus,
                                                    "memory_mb": phase.controls.load.fence.memory_mb,
                                                    "cpu_shares": phase.controls.load.fence.cpu_shares,
                                                }
                                                if phase.controls.load.fence is not None
                                                else None
                                            ),
                                            "in_container": (
                                                {
                                                    "cpu_workers": phase.controls.load.in_container.cpu_workers,
                                                    "stream_workers": phase.controls.load.in_container.stream_workers,
                                                    "vm_workers": phase.controls.load.in_container.vm_workers,
                                                    "vm_bytes_mb": phase.controls.load.in_container.vm_bytes_mb,
                                                }
                                                if phase.controls.load.in_container is not None
                                                else None
                                            ),
                                        }
                                        if phase.controls.load is not None
                                        else None
                                    ),
                                },
                            }
                            for phase in scenario.phases
                        ],
                    }
                    for scenario in self.runtime_stress.scenarios
                ],
            }

        return config_dict


def load_config(path: Union[str, Path]) -> Config:
    """
    Load and parse YAML configuration file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Parsed configuration object

    Raises:
        FileNotFoundError: If configuration file doesn't exist
        yaml.YAMLError: If YAML parsing fails
        ValueError: If configuration validation fails
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    logger.info(f"Loading configuration from {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML configuration: {e}")

    if not config_dict:
        raise ValueError("Configuration file is empty")

    # Parse each section
    try:
        experiment = parse_experiment(config_dict)
        dataset = parse_dataset(config_dict)
        perturbations = parse_perturbations(config_dict)
        output = parse_output(config_dict)
        robustness_boundary = parse_robustness_boundary(config_dict)
        runtime_stress = parse_runtime_stress(config_dict)
    except Exception as e:
        logger.error(f"Configuration parsing failed: {e}")
        raise

    config = Config(
        experiment=experiment,
        dataset=dataset,
        perturbations=perturbations,
        output=output,
        robustness_boundary=robustness_boundary,
        runtime_stress=runtime_stress,
    )

    # Parse optional sections
    if 'profiling' in config_dict:
        config.profiling = parse_profiling(config_dict)

    if 'slam' in config_dict:
        config.slam = parse_slam(config_dict)

    logger.info(f"Configuration loaded successfully: {config}")
    return config


def save_config(config: Config, path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration object to save
        path: Path to save YAML file
    """
    path = Path(path)

    logger.info(f"Saving configuration to {path}")

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    logger.info("Configuration saved successfully")
