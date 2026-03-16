"""YAML configuration parser for SLAMAdverserialLab."""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from ..utils import get_logger
from .schema import (
    ExperimentConfig,
    DatasetConfig,
    PerturbationConfig,
    OutputConfig,
    RobustnessBoundaryConfig,
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


class Config:
    """Complete configuration container."""

    def __init__(
        self,
        experiment: ExperimentConfig,
        dataset: DatasetConfig,
        perturbations: List[PerturbationConfig],
        output: OutputConfig,
        robustness_boundary: Optional[RobustnessBoundaryConfig] = None,
    ):
        """
        Initialize configuration.

        Args:
            experiment: Experiment configuration
            dataset: Dataset configuration
            perturbations: List of perturbation configurations
            output: Output configuration
            robustness_boundary: Optional robustness-boundary configuration
        """
        self.experiment = experiment
        self.dataset = dataset
        self.perturbations = perturbations
        self.output = output
        self.robustness_boundary = robustness_boundary

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
    except Exception as e:
        logger.error(f"Configuration parsing failed: {e}")
        raise

    config = Config(
        experiment=experiment,
        dataset=dataset,
        perturbations=perturbations,
        output=output,
        robustness_boundary=robustness_boundary,
    )

    # Parse optional sections
    if 'profiling' in config_dict:
        config.profiling = parse_profiling(config_dict)

    if 'slam' in config_dict:
        config.slam = parse_slam(config_dict)

    # TODO: Add validation when validation.py is created
    # from .validation import validate_config
    # validation_result = validate_config(config)
    # if not validation_result.valid:
    #     logger.error("Configuration validation failed")
    #     validation_result.print_summary()
    #     raise ValueError(f"Configuration has {len(validation_result.errors)} validation error(s)")
    # if validation_result.warnings:
    #     logger.warning(f"Configuration has {len(validation_result.warnings)} warning(s)")
    #     validation_result.print_summary()

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
