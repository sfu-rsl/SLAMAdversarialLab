"""Core pipeline components for SLAMAdverserialLab."""

from .pipeline import Pipeline
from .frame import Frame
from .output import (
    OutputFormat,
    OutputConfig as OutputSaveConfig,
    OutputWriter,
    ImageWriter,
    OutputManager,
    create_output_manager
)

__all__ = [
    'Pipeline',
    'Frame',
    'OutputFormat',
    'OutputSaveConfig',
    'OutputWriter',
    'ImageWriter',
    'OutputManager',
    'create_output_manager'
]