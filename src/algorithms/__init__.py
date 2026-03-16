"""SLAM algorithm implementations."""

from .base import SLAMAlgorithm
from .types import SensorMode, SLAMRunRequest, SLAMRunResult
from .registry import get_slam_algorithm, list_slam_algorithms, register_slam_algorithm

__all__ = [
    'SLAMAlgorithm',
    'SensorMode',
    'SLAMRunRequest',
    'SLAMRunResult',
    'get_slam_algorithm',
    'list_slam_algorithms',
    'register_slam_algorithm',
]

# Conditionally export algorithm classes if available
try:
    from .orbslam3 import ORBSLAM3Algorithm
    __all__.append('ORBSLAM3Algorithm')
except ImportError:
    pass

try:
    from .s3pogs import S3POGSAlgorithm
    __all__.append('S3POGSAlgorithm')
except ImportError:
    pass

try:
    from .pyslam_runner import PySLAMRunner
    __all__.append('PySLAMRunner')
except ImportError:
    pass
