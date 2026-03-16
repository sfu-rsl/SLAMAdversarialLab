"""
SLAMAdverserialLab: An Extensible Framework for Visual SLAM Robustness Evaluation under Adverse Conditions.
"""

from .__version__ import __version__
from .utils import get_logger

logger = get_logger(__name__)
logger.debug(f"SLAMAdverserialLab v{__version__} initialized")

__all__ = ['__version__', 'get_logger']
