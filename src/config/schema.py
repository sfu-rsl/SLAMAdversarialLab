"""Configuration schema definitions using dataclasses."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
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
