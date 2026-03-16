"""Common request/response types for SLAM algorithm plugins."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional


class SensorMode(str, Enum):
    """Sensor mode used for SLAM execution."""

    MONO = "mono"
    STEREO = "stereo"
    RGBD = "rgbd"


@dataclass(frozen=True)
class SLAMRunRequest:
    """Structured request passed to a SLAM algorithm plugin."""

    dataset_path: Path
    slam_config: str
    output_dir: Path
    dataset_type: str
    sensor_mode: SensorMode
    sequence_name: str
    extras: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.sequence_name).strip():
            raise ValueError(
                "SLAMRunRequest.sequence_name is required and must be non-empty."
            )


@dataclass
class SLAMRuntimeContext:
    """Per-run derived state shared across algorithm lifecycle hooks."""

    request: SLAMRunRequest
    config_is_external: bool
    resolved_config_path: Optional[Path]
    internal_config_name: Optional[str]
    sequence_name: str
    effective_dataset_path: Optional[Path] = None
    execution_inputs: Dict[str, Any] = field(default_factory=dict)
    staging_artifacts: Dict[str, Any] = field(default_factory=dict)
    notes: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SLAMRunResult:
    """Structured response returned by a SLAM algorithm plugin."""

    success: bool
    trajectory_path: Optional[Path] = None
    message: str = ""
