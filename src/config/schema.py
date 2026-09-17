"""Configuration schema definitions using dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    name: str
    description: str = ""
    version: str = "1.0.0"
    seed: Optional[int] = None

    def validate(self) -> None:
        """Validate experiment configuration."""
        if not self.name:
            raise ValueError("Experiment name is required")
        # Allow more characters but check for filesystem-unsafe ones
        invalid_chars = set('<>:"|?*/')
        if any(c in self.name for c in invalid_chars):
            raise ValueError(
                f"Experiment name '{self.name}' contains invalid filesystem characters. "
                f"Avoid: {' '.join(invalid_chars)}"
            )


@dataclass
class DatasetConfig:
    """Configuration for dataset loading.

    For TUM datasets, you can specify either:
    - sequence: Sequence name (e.g., "freiburg1_desk") - auto-downloads if missing
    - path: Explicit path to dataset directory

    For KITTI datasets:
    - sequence: Sequence number (e.g., "00", "04")
    - Path is auto-resolved to ./datasets/kitti/sequences/{sequence}

    Path resolution is handled by each dataset class's resolve_path() method.
    """

    type: str  # 'tum', 'kitti', 'mock', etc.
    path: Optional[str] = None  # Explicit path (optional if sequence is provided)
    sequence: Optional[str] = None  # Canonical sequence name/number (auto-resolves path)
    max_frames: Optional[int] = None  # Limit frames for testing
    skip_depth: bool = False  # Skip loading depth data (for datasets without depth)
    load_stereo: bool = False  # Load stereo image pairs (for KITTI)

    def validate(self) -> None:
        """Validate dataset configuration."""
        if not self.type:
            raise ValueError("Dataset type is required")

        from ..datasets.factory import list_datasets
        valid_types = list_datasets() + ["custom"]
        if self.type.lower() not in valid_types:
            raise ValueError(
                f"Invalid dataset type '{self.type}'. "
                f"Must be one of: {', '.join(valid_types)}"
            )

        # For mock datasets, path is not required
        if self.type == "mock":
            return

        # Must have either path or sequence
        if not self.path and not self.sequence:
            raise ValueError(
                f"Dataset requires either 'path' or 'sequence'. "
                f"For {self.type}, specify sequence name/number for auto-resolution."
            )


@dataclass
class PerturbationConfig:
    """Configuration for a perturbation module."""

    name: str
    type: str  # 'fog', 'rain', 'none', etc.
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate perturbation configuration."""
        if not self.name:
            raise ValueError("Perturbation name is required")

        if not self.type:
            raise ValueError("Perturbation type is required")

        # Import here to avoid circular imports
        from ..modules.base import get_module_registry
        registry = get_module_registry()

        # Built-in types that don't need registry lookup
        builtin_types = {"none", "composite"}

        if self.type not in builtin_types and self.type not in registry:
            available = sorted(registry.keys())
            raise ValueError(
                f"Invalid perturbation type '{self.type}'. "
                f"Available types: none, composite, {', '.join(available)}"
            )

        # Type-specific parameter validation
        if self.type == "fog":
            self._validate_fog_params()
        elif self.type == "rain":
            self._validate_rain_params()
        elif self.type == "composite":
            self._validate_composite_params()

    def _validate_fog_params(self) -> None:
        """Validate fog-specific parameters."""
        params = self.parameters

        if "preset" in params:
            preset = params["preset"]
            valid_presets = ["light", "medium", "heavy", "dense"]
            if preset not in valid_presets:
                raise ValueError(
                    f"Invalid fog preset '{preset}'. "
                    f"Must be one of: {', '.join(valid_presets)}"
                )
            # If preset is specified, check for conflicting params
            if "preset" in params and ("visibility_m" in params or "beta" in params):
                logger.warning(
                    "Fog preset specified along with visibility_m/beta. "
                    "Custom values will override preset."
                )

        if "visibility_m" in params and "beta" in params:
            raise ValueError(
                "Cannot specify both 'visibility_m' and 'beta'. Choose one."
            )

        if "visibility_m" in params:
            vis = params["visibility_m"]
            if not isinstance(vis, (int, float)) or vis <= 0:
                raise ValueError(
                    f"visibility_m must be a positive number, got {vis}"
                )
            # Warn if visibility is very low or very high
            if vis < 5:
                logger.warning(f"Very low visibility ({vis}m) may produce extreme fog")
            elif vis > 500:
                logger.warning(f"Very high visibility ({vis}m) will produce minimal fog")

        if "beta" in params:
            beta = params["beta"]
            if not isinstance(beta, (int, float)) or beta <= 0:
                raise ValueError(
                    f"beta must be a positive number, got {beta}"
                )

        if "atmospheric_light" in params:
            light = params["atmospheric_light"]
            if not isinstance(light, (list, tuple)) or len(light) != 3:
                raise ValueError(
                    "atmospheric_light must be a list/tuple of 3 values [R, G, B]"
                )
            for val in light:
                if not (0.0 <= val <= 1.0):
                    raise ValueError(
                        f"atmospheric_light values must be in [0, 1], got {val}"
                    )

        if "min_depth_m" in params:
            min_d = params["min_depth_m"]
            if not isinstance(min_d, (int, float)) or min_d < 0:
                raise ValueError(f"min_depth_m must be non-negative, got {min_d}")

        if "max_depth_m" in params:
            max_d = params["max_depth_m"]
            if not isinstance(max_d, (int, float)) or max_d <= 0:
                raise ValueError(f"max_depth_m must be positive, got {max_d}")

        if "min_depth_m" in params and "max_depth_m" in params:
            if params["min_depth_m"] >= params["max_depth_m"]:
                raise ValueError(
                    f"min_depth_m ({params['min_depth_m']}) must be less than "
                    f"max_depth_m ({params['max_depth_m']})"
                )

        if "noise_backend" in params:
            noise_backend = params["noise_backend"]
            if not isinstance(noise_backend, str) or not noise_backend.strip():
                raise ValueError("noise_backend must be a non-empty string")
            valid_noise_backends = {"auto", "simplex", "perlin"}
            if noise_backend.strip().lower() not in valid_noise_backends:
                raise ValueError(
                    f"Invalid noise_backend '{noise_backend}'. "
                    f"Must be one of: {', '.join(sorted(valid_noise_backends))}"
                )

        if "strict_simplex" in params:
            raise ValueError(
                "strict_simplex is no longer supported. "
                "Use noise_backend: simplex to require SimplexNoise."
            )

        self._validate_depth_backend_param()

    def _validate_rain_params(self) -> None:
        """Validate physics-based rain parameters."""
        params = self.parameters

        if "intensity" in params:
            intensity = params["intensity"]
            if not isinstance(intensity, (int, float)) or intensity < 1 or intensity > 200:
                raise ValueError(
                    f"Rain intensity must be between 1 and 200 mm/hr, got {intensity}"
                )

        if "depth_model" in params:
            model = params["depth_model"]
            valid_models = ["vits", "vitb", "vitl"]
            if model not in valid_models:
                raise ValueError(
                    f"Invalid depth model '{model}'. Must be one of: {', '.join(valid_models)}"
                )

        if "max_depth" in params:
            max_depth = params["max_depth"]
            if not isinstance(max_depth, (int, float)) or max_depth <= 0:
                raise ValueError(f"max_depth must be positive, got {max_depth}")

        self._validate_depth_backend_param()

    def _validate_depth_backend_param(self) -> None:
        """Validate optional explicit depth backend selection."""
        params = self.parameters
        if "depth_backend" not in params:
            return

        value = params["depth_backend"]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"depth_backend must be a non-empty string, got {value!r}"
            )

        normalized = value.strip().lower()
        valid = {"auto", "existing", "foundation_stereo", "da3", "da2"}
        if normalized not in valid:
            raise ValueError(
                f"Invalid depth_backend '{value}'. Must be one of: {', '.join(sorted(valid))}"
            )

    def _validate_composite_params(self) -> None:
        """Validate composite module parameters."""
        params = self.parameters

        if "modules" not in params:
            raise ValueError("Composite module requires 'modules' parameter")

        modules = params["modules"]
        if not isinstance(modules, list):
            raise ValueError("'modules' parameter must be a list")

        if not modules:
            raise ValueError("'modules' list cannot be empty")

        for i, module in enumerate(modules):
            if not isinstance(module, dict):
                raise ValueError(f"Module {i} must be a dictionary configuration")

            if "type" not in module:
                raise ValueError(f"Module {i} missing required 'type' field")

        if "mode" in params:
            mode = params["mode"]
            valid_modes = ["sequential"]
            if mode not in valid_modes:
                raise ValueError(
                    f"Invalid composition mode '{mode}'. "
                    f"Must be one of: {', '.join(valid_modes)}"
                )


@dataclass
class RobustnessBoundaryConfig:
    """Configuration for robustness-boundary search.

    Boundary classification is based on the ATE threshold and, optionally,
    whether tracking failure itself should count as a failed trial.
    """

    enabled: bool = False
    name: str = ""
    target_perturbation: str = ""
    module: str = ""
    parameter: str = ""
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    tolerance: float = 0.05
    max_iters: int = 8
    # Trials with mean ATE RMSE above this threshold are classified as failed.
    ate_rmse_fail: float = 1.5
    # When True, missing trajectories / tracking loss are treated as failed trials.
    # When False, tracking failure is recorded but not fatal by itself.
    fail_on_tracking_failure: bool = True

    def validate(self) -> None:
        """Validate robustness-boundary configuration."""
        if not isinstance(self.enabled, bool):
            raise ValueError(f"robustness_boundary.enabled must be boolean, got {self.enabled!r}")

        # Disabled block is allowed to omit fields.
        if not self.enabled:
            return

        if not isinstance(self.name, str):
            raise ValueError(f"robustness_boundary.name must be a string, got {self.name!r}")

        if not isinstance(self.target_perturbation, str):
            raise ValueError(
                "robustness_boundary.target_perturbation must be a string, "
                f"got {self.target_perturbation!r}"
            )

        if self.name:
            invalid_chars = set('<>:"|?*/\\')
            if any(c in self.name for c in invalid_chars):
                raise ValueError(
                    f"robustness_boundary.name '{self.name}' contains invalid filesystem characters. "
                    f"Avoid: {' '.join(invalid_chars)}"
                )

        if not self.module:
            raise ValueError("robustness_boundary.module is required when enabled")

        if not self.parameter:
            raise ValueError("robustness_boundary.parameter is required when enabled")

        if self.lower_bound is None:
            raise ValueError("robustness_boundary.lower_bound is required when enabled")

        if self.upper_bound is None:
            raise ValueError("robustness_boundary.upper_bound is required when enabled")

        if not isinstance(self.max_iters, int) or isinstance(self.max_iters, bool) or self.max_iters < 1:
            raise ValueError(
                f"robustness_boundary.max_iters must be an integer >= 1, got {self.max_iters!r}"
            )

        if not isinstance(self.tolerance, (int, float)) or isinstance(self.tolerance, bool) or self.tolerance <= 0:
            raise ValueError(
                f"robustness_boundary.tolerance must be a positive number, got {self.tolerance!r}"
            )

        if (
            not isinstance(self.ate_rmse_fail, (int, float))
            or isinstance(self.ate_rmse_fail, bool)
            or self.ate_rmse_fail <= 0
        ):
            raise ValueError(
                f"robustness_boundary.ate_rmse_fail must be a positive number, got {self.ate_rmse_fail!r}"
            )

        if not isinstance(self.fail_on_tracking_failure, bool):
            raise ValueError(
                "robustness_boundary.fail_on_tracking_failure must be boolean, "
                f"got {self.fail_on_tracking_failure!r}"
            )

        from ..modules.base import get_module_registry
        from ..robustness.param_spec import parse_domain_value

        registry = get_module_registry()
        if self.module not in registry:
            available = sorted(
                name
                for name, reg in registry.items()
                if getattr(reg.module_class, "SEARCHABLE_PARAMS", {})
            )
            raise ValueError(
                f"Unknown robustness_boundary.module '{self.module}'. "
                f"Available boundary-enabled modules: {', '.join(available) if available else '(none)'}"
            )

        module_class = registry[self.module].module_class
        searchable_params = getattr(module_class, "SEARCHABLE_PARAMS", {})
        if not searchable_params:
            raise ValueError(
                f"Module '{self.module}' does not declare any boundary-search parameters "
                f"(SEARCHABLE_PARAMS is empty)."
            )

        if self.parameter not in searchable_params:
            available = sorted(searchable_params.keys())
            raise ValueError(
                f"Parameter '{self.parameter}' is not supported for robustness boundary in module '{self.module}'. "
                f"Supported parameters: {', '.join(available)}"
            )

        spec = searchable_params[self.parameter]
        try:
            lower_value = parse_domain_value(spec, self.lower_bound)
        except ValueError as exc:
            raise ValueError(
                f"Invalid robustness_boundary.lower_bound for {self.module}.{self.parameter}: {exc}"
            ) from exc

        try:
            upper_value = parse_domain_value(spec, self.upper_bound)
        except ValueError as exc:
            raise ValueError(
                f"Invalid robustness_boundary.upper_bound for {self.module}.{self.parameter}: {exc}"
            ) from exc

        if lower_value >= upper_value:
            raise ValueError(
                f"robustness_boundary.lower_bound must be less than upper_bound "
                f"(got {self.lower_bound!r} >= {self.upper_bound!r})"
            )


@dataclass
class RuntimeStressTelemetryConfig:
    """Configuration for runtime-stress telemetry sampling."""

    sample_period_ms: int = 500

    def validate(self) -> None:
        """Validate telemetry configuration."""
        if (
            not isinstance(self.sample_period_ms, int)
            or isinstance(self.sample_period_ms, bool)
            or self.sample_period_ms <= 0
        ):
            raise ValueError(
                "runtime_stress.telemetry.sample_period_ms must be a positive integer"
            )


@dataclass
class CpuControlConfig:
    """CPU runtime-stress controls."""

    max_cores: Optional[float] = None

    def validate(self) -> None:
        """Validate CPU control configuration."""
        if self.max_cores is None:
            return

        if (
            not isinstance(self.max_cores, (int, float))
            or isinstance(self.max_cores, bool)
            or self.max_cores <= 0
        ):
            raise ValueError(
                "runtime_stress.controls.cpu.max_cores must be a positive number"
            )


@dataclass
class MemoryControlConfig:
    """Memory runtime-stress controls."""

    max_mb: Optional[int] = None

    def validate(self) -> None:
        """Validate memory control configuration."""
        if self.max_mb is None:
            return

        if (
            not isinstance(self.max_mb, int)
            or isinstance(self.max_mb, bool)
            or self.max_mb <= 0
        ):
            raise ValueError(
                "runtime_stress.controls.memory.max_mb must be a positive integer"
            )


@dataclass
class GpuControlConfig:
    """GPU runtime-stress controls backed by HAMi-core."""

    vram_limit_mb: Optional[int] = None
    sm_limit_percent: Optional[int] = None

    def validate(self) -> None:
        """Validate GPU control configuration."""
        if self.vram_limit_mb is not None:
            if (
                not isinstance(self.vram_limit_mb, int)
                or isinstance(self.vram_limit_mb, bool)
                or self.vram_limit_mb <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.gpu.vram_limit_mb must be a positive integer"
                )

        if self.sm_limit_percent is not None:
            if (
                not isinstance(self.sm_limit_percent, int)
                or isinstance(self.sm_limit_percent, bool)
                or self.sm_limit_percent <= 0
                or self.sm_limit_percent > 100
            ):
                raise ValueError(
                    "runtime_stress.controls.gpu.sm_limit_percent must be an integer in (0, 100]"
                )


@dataclass
class IoControlConfig:
    """Block-IO runtime-stress controls (bandwidth and IOPS caps)."""

    read_bps: Optional[int] = None
    write_bps: Optional[int] = None
    read_iops: Optional[int] = None
    write_iops: Optional[int] = None

    def validate(self) -> None:
        """Validate IO control configuration."""
        for field_name in ("read_bps", "write_bps", "read_iops", "write_iops"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(
                    f"runtime_stress.controls.io.{field_name} must be a positive integer"
                )


@dataclass
class LoadGpuAntagonistConfig:
    """GPU load-antagonist spec (a contending CUDA workload, not a cap)."""

    vram_mb: Optional[int] = None
    matmul_n: Optional[int] = None
    duty_cycle: Optional[float] = None
    image: Optional[str] = None

    def validate(self) -> None:
        """Validate GPU antagonist configuration."""
        if self.vram_mb is not None:
            if (
                not isinstance(self.vram_mb, int)
                or isinstance(self.vram_mb, bool)
                or self.vram_mb < 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.gpu.vram_mb must be a non-negative integer"
                )
        if self.duty_cycle is not None:
            if (
                not isinstance(self.duty_cycle, (int, float))
                or isinstance(self.duty_cycle, bool)
                or self.duty_cycle < 0
                or self.duty_cycle > 1
            ):
                raise ValueError(
                    "runtime_stress.controls.load.gpu.duty_cycle must be a number in [0, 1]"
                )
        if self.duty_cycle is not None and self.duty_cycle > 0:
            if (
                not isinstance(self.matmul_n, int)
                or isinstance(self.matmul_n, bool)
                or self.matmul_n <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.gpu.matmul_n must be a positive integer "
                    "when duty_cycle > 0"
                )
        if self.image is not None and not isinstance(self.image, str):
            raise ValueError("runtime_stress.controls.load.gpu.image must be a string")
        vram_active = self.vram_mb is not None and self.vram_mb > 0
        duty_active = self.duty_cycle is not None and self.duty_cycle > 0
        if not vram_active and not duty_active:
            raise ValueError(
                "runtime_stress.controls.load.gpu must set vram_mb > 0 and/or duty_cycle > 0"
            )


@dataclass
class LoadFenceConfig:
    """DEPRECATED (sibling-antagonist calibration). Prefer `in_container`.

    Calibration applied to the SIBLING stress-ng antagonist container. Both
    modes below are deprecated as of the in-container migration: the
    in-container mode (`LoadInContainerConfig`) is the supported way to apply
    CPU-side contention. These are kept working, not removed, because
    Conditions C8/C9/C11/C12/C14/C15 were measured with them and those
    results are part of this project's published experiment record.
    (The GPU antagonist is also a sibling container but is NOT deprecated --
    it is the only GPU-load mechanism.)

    Calibration applied to the antagonist container itself.

    Two calibration modes, both making load-based pressure reproducible:
    - ``cpus`` (quota): the antagonist consumes at most this many cores. Its
      fair-share cgroup still lets the SLAM keep its full demand, so this
      models an ORCHESTRATED host where the SLAM is a protected peer.
    - ``cpu_shares`` (weight): the antagonist runs UNCAPPED but with this CPU
      weight (default 1024). A weight far above the SLAM's collapses the
      SLAM's proportional fair-share slice below its demand, starving it.
      This models an UNMANAGED/oversubscribed host where the SLAM is
      best-effort (a nice-19 SLAM is ~a 68x weight disadvantage).
    At least one of ``cpus``/``cpu_shares`` must be set (enforced in
    LoadControlConfig).
    """

    cpus: Optional[float] = None
    memory_mb: Optional[int] = None
    cpu_shares: Optional[int] = None

    # podman --cpu-shares accepted range (cgroup v1 units, mapped to v2 weight)
    _MIN_SHARES = 2
    _MAX_SHARES = 262144

    def validate(self) -> None:
        """Validate fence configuration."""
        if self.cpus is not None:
            if (
                not isinstance(self.cpus, (int, float))
                or isinstance(self.cpus, bool)
                or self.cpus <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.fence.cpus must be a positive number"
                )
        if self.memory_mb is not None:
            if (
                not isinstance(self.memory_mb, int)
                or isinstance(self.memory_mb, bool)
                or self.memory_mb <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.fence.memory_mb must be a positive integer"
                )
        if self.cpu_shares is not None:
            if (
                not isinstance(self.cpu_shares, int)
                or isinstance(self.cpu_shares, bool)
                or self.cpu_shares < self._MIN_SHARES
                or self.cpu_shares > self._MAX_SHARES
            ):
                raise ValueError(
                    "runtime_stress.controls.load.fence.cpu_shares must be an integer in "
                    f"[{self._MIN_SHARES}, {self._MAX_SHARES}]"
                )


@dataclass
class LoadInContainerConfig:
    """In-container load: stress-ng workers exec'd INTO the SLAM's own container.

    Unlike the sibling antagonists (separate cgroups), these run in the SLAM's
    cgroup, so they compete flat/thread-for-thread at EQUAL priority (no fence,
    no weight). Models a monolithic stack or a bare OS with no per-workload
    isolation. Backed by the framework's static stress-ng binary, bind-mounted
    at launch (deps/stress-ng/) -- the same generator as the sibling modes, so
    only the PLACEMENT varies across modes. Severity for the CPU crowd =
    ``cpu_workers``: the SLAM gets slam_threads/(slam_threads + cpu_workers)
    of the machine once the cgroup saturates.
    """

    cpu_workers: Optional[int] = None
    stream_workers: Optional[int] = None
    vm_workers: Optional[int] = None
    vm_bytes_mb: Optional[int] = None

    _MAX_CPU_WORKERS = 1024  # crowd semantics need > the sibling cap of 32
    _MAX_OTHER_WORKERS = 64
    _MAX_VM_TOTAL_MB = 32768  # hard safety bound: ballast lands in the SLAM's
    # (usually uncapped) cgroup, so total vm must never approach host RAM

    def validate(self) -> None:
        if self.cpu_workers is not None:
            if (
                not isinstance(self.cpu_workers, int)
                or isinstance(self.cpu_workers, bool)
                or self.cpu_workers <= 0
                or self.cpu_workers > self._MAX_CPU_WORKERS
            ):
                raise ValueError(
                    "runtime_stress.controls.load.in_container.cpu_workers must be an "
                    f"integer in [1, {self._MAX_CPU_WORKERS}]"
                )
        for field_name in ("stream_workers", "vm_workers"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > self._MAX_OTHER_WORKERS
            ):
                raise ValueError(
                    f"runtime_stress.controls.load.in_container.{field_name} must be "
                    f"an integer in [1, {self._MAX_OTHER_WORKERS}]"
                )
        if self.vm_bytes_mb is not None:
            if (
                not isinstance(self.vm_bytes_mb, int)
                or isinstance(self.vm_bytes_mb, bool)
                or self.vm_bytes_mb <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.in_container.vm_bytes_mb must be a "
                    "positive integer"
                )
        if (self.vm_workers is None) != (self.vm_bytes_mb is None):
            raise ValueError(
                "runtime_stress.controls.load.in_container.vm_workers and vm_bytes_mb "
                "must be set together (both or neither)"
            )
        if self.vm_workers and self.vm_bytes_mb:
            total = self.vm_workers * self.vm_bytes_mb
            if total > self._MAX_VM_TOTAL_MB:
                raise ValueError(
                    "runtime_stress.controls.load.in_container vm total "
                    f"(vm_workers x vm_bytes_mb = {total} MB) exceeds the "
                    f"{self._MAX_VM_TOTAL_MB} MB safety bound: the ballast lands in "
                    "the SLAM's own (usually uncapped) cgroup and must never "
                    "approach host RAM"
                )
        if not any((self.cpu_workers, self.stream_workers, self.vm_workers)):
            raise ValueError(
                "runtime_stress.controls.load.in_container must set at least one of "
                "cpu_workers / stream_workers / vm_workers"
            )


@dataclass
class LoadSegmentationConfig:
    """SAM 3 segmentation running beside the SLAM: a REAL co-tenant workload.

    The other antagonists are synthetic. This one is what a robot actually runs
    next to SLAM -- labelling the map downstream, masking dynamic objects
    upstream -- so a reviewer cannot answer it with "nobody deploys stress-ng".

    NO SEVERITY KNOB, BY MEASUREMENT rather than by omission. One instance flat
    out holds 88-100% SM on a 3090, so it is the ceiling and not a rung: a second
    instance adds no compute pressure and only doubles VRAM. The treatment is
    binary, segmentation running or not, and pretending otherwise would invent a
    ladder the hardware cannot deliver.

    THE FRAME SOURCE IS DELIBERATELY NOT COUPLED TO THE SLAM'S DATASET BLOCK.
    `source: dataset` copies four things and nothing else -- type, path/sequence,
    max_frames, load_stereo. It is a DATA reference, not a runtime one. The
    segmenter does NOT share the SLAM's deadline, its frame timing, its warmup or
    its phases; it reads the same files at its own pace. "Same dataset" means the
    same images and the same frame budget, which is the realistic part (one
    camera feeding two consumers), without inventing a synchronisation the
    experiment does not actually have.
    """

    frames_dir: Optional[str] = None
    source: Optional[str] = None          # "dataset" -> inherit the dataset block
    prompt: str = "person"
    max_frames: Optional[int] = None
    conda_env: str = "sam3"

    def validate(self) -> None:
        if self.source is not None and self.source != "dataset":
            raise ValueError(
                "runtime_stress.controls.load.segmentation.source must be "
                "'dataset' (inherit the experiment's dataset block) or omitted "
                "in favour of an explicit frames_dir"
            )
        if self.source is None and not self.frames_dir:
            raise ValueError(
                "runtime_stress.controls.load.segmentation needs either "
                "source: dataset or an explicit frames_dir -- refusing to guess "
                "which images the co-tenant should segment"
            )
        if self.source is not None and self.frames_dir:
            raise ValueError(
                "runtime_stress.controls.load.segmentation sets BOTH source and "
                "frames_dir; they are alternatives and having both hides which "
                "one actually took effect"
            )
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError(
                "runtime_stress.controls.load.segmentation.prompt must be a "
                "non-empty string (SAM 3 is promptable with text; the prompt "
                "names the use case, e.g. 'person' for dynamic-object masking)"
            )
        if self.max_frames is not None:
            if (
                not isinstance(self.max_frames, int)
                or isinstance(self.max_frames, bool)
                or self.max_frames <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.segmentation.max_frames must "
                    "be a positive integer"
                )
        if not isinstance(self.conda_env, str) or not self.conda_env.strip():
            raise ValueError(
                "runtime_stress.controls.load.segmentation.conda_env must be a "
                "non-empty environment name"
            )


@dataclass
class LoadControlConfig:
    """Load-antagonist controls: spawn contending workloads during the phase.

    Unlike the cap axes (which shrink the SLAM's own allocation), load
    antagonists CONTEND with the SLAM. Sibling antagonists (stress-ng, gpu)
    run in their own fenced containers; in_container runs inside the SLAM's
    cgroup for flat equal-priority competition.
    """

    # DEPRECATED sibling stress-ng knobs (with `fence`): use `in_container`
    # instead. Kept working so the C8/C9/C11/C12/C14/C15 results stay
    # reproducible; a runtime warning fires when they are used.
    cpu_workers: Optional[int] = None
    stream_workers: Optional[int] = None
    vm_workers: Optional[int] = None
    vm_bytes_mb: Optional[int] = None
    fence: Optional[LoadFenceConfig] = None
    # Supported paths:
    gpu: Optional[LoadGpuAntagonistConfig] = None          # GPU load (sibling, NOT deprecated)
    in_container: Optional[LoadInContainerConfig] = None   # CPU-side load, preferred
    segmentation: Optional[LoadSegmentationConfig] = None  # SAM 3 co-tenant, GPU-side

    _MAX_WORKERS = 32

    def validate(self) -> None:
        """Validate load-antagonist configuration (fail loud, no silent defaults)."""
        for field_name in ("cpu_workers", "stream_workers", "vm_workers"):
            value = getattr(self, field_name)
            if value is None:
                continue
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > self._MAX_WORKERS
            ):
                raise ValueError(
                    f"runtime_stress.controls.load.{field_name} must be an integer in "
                    f"[1, {self._MAX_WORKERS}]"
                )
        if self.vm_bytes_mb is not None:
            if (
                not isinstance(self.vm_bytes_mb, int)
                or isinstance(self.vm_bytes_mb, bool)
                or self.vm_bytes_mb <= 0
            ):
                raise ValueError(
                    "runtime_stress.controls.load.vm_bytes_mb must be a positive integer"
                )
        if (self.vm_workers is None) != (self.vm_bytes_mb is None):
            raise ValueError(
                "runtime_stress.controls.load.vm_workers and vm_bytes_mb must be set "
                "together (both or neither)"
            )
        if self.gpu is not None:
            self.gpu.validate()
        if self.fence is not None:
            self.fence.validate()
        if self.in_container is not None:
            self.in_container.validate()
        if self.segmentation is not None:
            self.segmentation.validate()

        stress_ng_active = any(
            getattr(self, f) for f in ("cpu_workers", "stream_workers", "vm_workers")
        )
        if (
            not stress_ng_active
            and self.gpu is None
            and self.in_container is None
            and self.segmentation is None
        ):
            raise ValueError(
                "runtime_stress.controls.load must request at least one antagonist "
                "(cpu_workers / stream_workers / vm_workers / gpu / in_container / "
                "segmentation)"
            )
        if stress_ng_active:
            if self.fence is None or (
                self.fence.cpus is None and self.fence.cpu_shares is None
            ):
                raise ValueError(
                    "runtime_stress.controls.load.fence must set cpus (quota) OR "
                    "cpu_shares (weight) whenever stress-ng workers are set: an "
                    "antagonist that is both uncapped AND unweighted is unbounded "
                    "and unreproducible"
                )
            if self.vm_workers and (self.fence.memory_mb is None):
                raise ValueError(
                    "runtime_stress.controls.load.fence.memory_mb is required when "
                    "vm_workers is set: an unfenced memory ballast can trigger the "
                    "host OOM killer and corrupt the whole cell"
                )


@dataclass
class RuntimeStressControlsConfig:
    """Controls applied during one runtime-stress phase."""

    cpu: Optional[CpuControlConfig] = None
    memory: Optional[MemoryControlConfig] = None
    gpu: Optional[GpuControlConfig] = None
    io: Optional[IoControlConfig] = None
    load: Optional[LoadControlConfig] = None

    def validate(self) -> None:
        """Validate phase controls."""
        if self.cpu is not None:
            self.cpu.validate()
        if self.memory is not None:
            self.memory.validate()
        if self.gpu is not None:
            self.gpu.validate()
        if self.io is not None:
            self.io.validate()
        if self.load is not None:
            self.load.validate()


@dataclass
class StressorPresetConfig:
    """A named, reusable multi-axis stress profile.

    Setting any subset of cpu/memory/gpu/io is allowed. The same axis
    config types and validators as RuntimeStressControlsConfig are reused
    so a preset's per-axis values are validated identically to inline
    phase controls.
    """

    cpu: Optional[CpuControlConfig] = None
    memory: Optional[MemoryControlConfig] = None
    gpu: Optional[GpuControlConfig] = None
    io: Optional[IoControlConfig] = None

    def validate(self) -> None:
        """Validate stressor preset configuration."""
        if (
            self.cpu is None
            and self.memory is None
            and self.gpu is None
            and self.io is None
        ):
            raise ValueError(
                "runtime_stress.stressors[*] must declare at least one of cpu/memory/gpu/io"
            )
        if self.cpu is not None:
            self.cpu.validate()
        if self.memory is not None:
            self.memory.validate()
        if self.gpu is not None:
            self.gpu.validate()
        if self.io is not None:
            self.io.validate()


@dataclass
class RuntimeStressPhaseConfig:
    """Configuration for one runtime-stress phase.

    A phase boundary is anchored to EITHER wall-clock seconds
    (``duration_s``) or the SLAM's processed-frame count (``until_frame``,
    the sampled-stream index at which this phase ends). Frame anchoring
    requires the scenario's ``realtime`` deadline harness, which reports
    the live frame number; it lets a control change land at the same
    point in the trajectory regardless of the paced frame rate. Exactly
    one of the two must be set per phase.
    """

    name: str
    duration_s: Optional[float] = None
    until_frame: Optional[int] = None
    controls: RuntimeStressControlsConfig = field(default_factory=RuntimeStressControlsConfig)
    stressors: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """Validate runtime-stress phase configuration."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime_stress.scenarios[].phases[].name is required")

        has_time = self.duration_s is not None
        has_frame = self.until_frame is not None
        if has_time and has_frame:
            raise ValueError(
                f"runtime_stress phase '{self.name}' sets both duration_s and "
                "until_frame; use exactly one"
            )
        if not has_time and not has_frame:
            raise ValueError(
                f"runtime_stress phase '{self.name}' must set duration_s or until_frame"
            )

        if has_time and (
            not isinstance(self.duration_s, (int, float))
            or isinstance(self.duration_s, bool)
            or self.duration_s <= 0
        ):
            raise ValueError(
                "runtime_stress.scenarios[].phases[].duration_s must be a positive number"
            )

        if has_frame and (
            not isinstance(self.until_frame, int)
            or isinstance(self.until_frame, bool)
            or self.until_frame <= 0
        ):
            raise ValueError(
                "runtime_stress.scenarios[].phases[].until_frame must be a positive integer"
            )

        if not isinstance(self.stressors, list):
            raise ValueError(
                "runtime_stress.scenarios[].phases[].stressors must be a list of names"
            )
        for ref in self.stressors:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(
                    "runtime_stress.scenarios[].phases[].stressors entries must be non-empty strings"
                )

        self.controls.validate()


@dataclass
class RealtimeDeadlineConfig:
    """Real-time frame-delivery deadline applied to a scenario.

    When set on a scenario, the runtime-stress pipeline injects a
    DeadlineIterator into the SLAM's frame loop (via the
    ``SAL_DEADLINE_FPS`` env var). Frames whose wall-clock deadline has
    passed are silently skipped, modeling drone/robot deployments where
    late frames get dropped instead of buffered.

    ``warmup_frames`` lets the SLAM consume the first N frames un-
    deadlined so per-SLAM init costs (CUDA kernel JIT, model load,
    lazy allocator setup) don't look like late frames. Counted in
    sampled-stream items (post-stride), not raw dataset frames.

    ``queue_size`` is the depth of the bounded FIFO buffer between the
    simulated camera and the SLAM (ROS-style ``queue_size``). Frames
    arrive at ``target_fps`` and the SLAM consumes the oldest still
    buffered; the oldest is dropped only on overflow. Default 1
    reproduces the original 1-deep "always freshest" behavior. Larger
    values absorb transient stalls (fewer drops) at the cost of the
    SLAM processing staler frames.
    """

    target_fps: float
    warmup_frames: int = 0
    queue_size: int = 1
    drop_policy: str = "drop_oldest"

    def validate(self) -> None:
        """Validate realtime deadline configuration."""
        if (
            not isinstance(self.target_fps, (int, float))
            or isinstance(self.target_fps, bool)
            or self.target_fps <= 0
        ):
            raise ValueError(
                "runtime_stress.scenarios[].realtime.target_fps must be a positive number"
            )

        if (
            not isinstance(self.warmup_frames, int)
            or isinstance(self.warmup_frames, bool)
            or self.warmup_frames < 0
        ):
            raise ValueError(
                "runtime_stress.scenarios[].realtime.warmup_frames must be a non-negative integer"
            )

        if (
            not isinstance(self.queue_size, int)
            or isinstance(self.queue_size, bool)
            or self.queue_size < 1
        ):
            raise ValueError(
                "runtime_stress.scenarios[].realtime.queue_size must be an integer >= 1"
            )

        if self.drop_policy not in ("drop_oldest", "drop_newest"):
            raise ValueError(
                "runtime_stress.scenarios[].realtime.drop_policy must be "
                "'drop_oldest' or 'drop_newest'"
            )


@dataclass
class RuntimeStressScenarioConfig:
    """Configuration for one named runtime-stress scenario."""

    name: str
    enabled: bool = True
    phases: List[RuntimeStressPhaseConfig] = field(default_factory=list)
    realtime: Optional[RealtimeDeadlineConfig] = None

    def validate(self) -> None:
        """Validate runtime-stress scenario configuration.

        Per-phase basic validation only. Cross-phase GPU consistency and
        stressor reference resolution are enforced at the
        ``RuntimeStressConfig`` level, which has access to the top-level
        stressor library.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("runtime_stress.scenarios[].name is required")

        if not isinstance(self.enabled, bool):
            raise ValueError("runtime_stress.scenarios[].enabled must be boolean")

        if not isinstance(self.phases, list) or not self.phases:
            raise ValueError(
                f"runtime_stress scenario '{self.name}' must declare at least one phase"
            )

        phase_names = set()
        for phase in self.phases:
            phase.validate()
            normalized_name = phase.name.strip()
            if normalized_name in phase_names:
                raise ValueError(
                    f"runtime_stress scenario '{self.name}' has duplicate phase name '{normalized_name}'"
                )
            phase_names.add(normalized_name)

        if self.realtime is not None:
            self.realtime.validate()

        # Frame-anchored phases: a scenario must be all-time or all-frame
        # (no mixing), must run under the realtime harness (it supplies the
        # live frame count), and the per-phase boundaries must be strictly
        # increasing so phase selection is unambiguous.
        frame_phases = [p for p in self.phases if p.until_frame is not None]
        if frame_phases:
            if len(frame_phases) != len(self.phases):
                raise ValueError(
                    f"runtime_stress scenario '{self.name}' mixes duration_s and "
                    "until_frame phases; a scenario must be all time-anchored or "
                    "all frame-anchored"
                )
            if self.realtime is None:
                raise ValueError(
                    f"runtime_stress scenario '{self.name}' uses until_frame phases "
                    "but has no realtime block; frame anchoring needs the deadline "
                    "harness to report the live frame count"
                )
            boundaries = [p.until_frame for p in self.phases]
            if any(b <= a for a, b in zip(boundaries[:-1], boundaries[1:])):
                raise ValueError(
                    f"runtime_stress scenario '{self.name}' until_frame boundaries "
                    f"must be strictly increasing, got {boundaries}"
                )


@dataclass
class RuntimeStressConfig:
    """Configuration for runtime-stress evaluation scenarios."""

    enabled: bool = False
    container_runtime: str = "docker"
    evict_page_cache: bool = False
    telemetry: RuntimeStressTelemetryConfig = field(default_factory=RuntimeStressTelemetryConfig)
    stressors: Dict[str, StressorPresetConfig] = field(default_factory=dict)
    scenarios: List[RuntimeStressScenarioConfig] = field(default_factory=list)

    def validate(self) -> None:
        """Validate runtime-stress configuration."""
        if not isinstance(self.enabled, bool):
            raise ValueError("runtime_stress.enabled must be boolean")

        if self.container_runtime not in {"docker", "podman"}:
            raise ValueError(
                f"runtime_stress.container_runtime must be 'docker' or 'podman', got {self.container_runtime!r}"
            )

        if not isinstance(self.evict_page_cache, bool):
            raise ValueError("runtime_stress.evict_page_cache must be boolean")

        if not self.enabled:
            return

        self.telemetry.validate()

        if not isinstance(self.stressors, dict):
            raise ValueError("runtime_stress.stressors must be a mapping of name -> preset")

        for preset_name, preset in self.stressors.items():
            if not isinstance(preset_name, str) or not preset_name.strip():
                raise ValueError(
                    "runtime_stress.stressors keys must be non-empty strings"
                )
            try:
                preset.validate()
            except ValueError as exc:
                raise ValueError(
                    f"runtime_stress.stressors['{preset_name}']: {exc}"
                ) from None

        if not isinstance(self.scenarios, list) or not self.scenarios:
            raise ValueError("runtime_stress.scenarios must contain at least one scenario when enabled")

        scenario_names = set()
        enabled_count = 0
        for scenario in self.scenarios:
            scenario.validate()
            self._validate_scenario_stressor_refs(scenario)
            normalized_name = scenario.name.strip()
            if normalized_name in scenario_names:
                raise ValueError(
                    f"runtime_stress scenario names must be unique; duplicate '{normalized_name}'"
                )
            scenario_names.add(normalized_name)
            if scenario.enabled:
                enabled_count += 1

        if enabled_count == 0:
            raise ValueError("runtime_stress.enabled=true requires at least one enabled scenario")

    def _validate_scenario_stressor_refs(self, scenario: "RuntimeStressScenarioConfig") -> None:
        """Reject phase references to undeclared stressor names."""
        for phase in scenario.phases:
            for ref in phase.stressors:
                if ref not in self.stressors:
                    raise ValueError(
                        f"runtime_stress scenario '{scenario.name}' phase "
                        f"'{phase.name}' references stressor '{ref}' which is "
                        "not declared in runtime_stress.stressors"
                    )


@dataclass
class OutputConfig:
    """Configuration for output handling."""

    base_dir: str = "./results"
    save_images: bool = True
    create_timestamp_dir: bool = True

    def validate(self) -> None:
        """Validate output configuration."""
        if not self.base_dir:
            raise ValueError("Output base_dir is required")

        base_path = Path(self.base_dir)
        try:
            base_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Cannot create output directory: {e}")

        test_file = base_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValueError(
                f"No write permission for output directory '{self.base_dir}': {e}"
            )


@dataclass
class ProfilingConfig:
    """Configuration for simple timing profiler."""

    enabled: bool = False
    verbose: bool = False
    save_report: bool = True
    report_format: str = "json"  # 'json' or 'txt'
    report_path: Optional[str] = None  # Will use output_dir/profiling if None

    def validate(self) -> None:
        """Validate profiling configuration."""
        valid_formats = ["json", "txt"]
        if self.report_format not in valid_formats:
            raise ValueError(
                f"Invalid report format '{self.report_format}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )
