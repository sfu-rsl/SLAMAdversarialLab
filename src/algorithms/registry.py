"""SLAM algorithm registry for dynamic algorithm selection."""

from typing import Dict, List, Type

from .base import SLAMAlgorithm


# Global registry mapping algorithm names to their classes
_SLAM_ALGORITHMS: Dict[str, Type[SLAMAlgorithm]] = {}


def register_slam_algorithm(name: str, algorithm_class: Type[SLAMAlgorithm]) -> None:
    """Register a SLAM algorithm implementation.

    Args:
        name: Algorithm identifier (e.g., 'orbslam3', 'vins')
        algorithm_class: Class implementing SLAMAlgorithm interface
    """
    _SLAM_ALGORITHMS[name] = algorithm_class


def get_slam_algorithm(name: str) -> SLAMAlgorithm:
    """Get a SLAM algorithm instance by name.

    Args:
        name: Algorithm identifier

    Returns:
        Instance of the requested SLAM algorithm

    Raises:
        ValueError: If algorithm name is not registered
    """
    if name not in _SLAM_ALGORITHMS:
        available = ', '.join(_SLAM_ALGORITHMS.keys()) if _SLAM_ALGORITHMS else 'none'
        raise ValueError(
            f"Unknown SLAM algorithm: '{name}'. "
            f"Available algorithms: {available}"
        )

    return _SLAM_ALGORITHMS[name]()


def list_slam_algorithms() -> List[str]:
    """Get list of all registered SLAM algorithms.

    Returns:
        List of algorithm names
    """
    return list(_SLAM_ALGORITHMS.keys())


def list_slam_algorithms_detailed() -> Dict[str, dict]:
    """Get detailed information about all registered SLAM algorithms.

    Returns:
        Dict mapping algorithm names to their info:
        {
            'name': str,
            'description': str,
            'supported_datasets': Dict[str, List[str]],  # e.g., {"kitti": ["mono", "stereo"]}
            'class_name': str
        }
    """
    result = {}
    for name, algorithm_class in _SLAM_ALGORITHMS.items():
        try:
            instance = algorithm_class()

            description = algorithm_class.__doc__
            if description:
                # Take first line of docstring
                description = description.strip().split('\n')[0]
            else:
                description = 'No description available'

            result[name] = {
                'name': instance.name,
                'description': description,
                'supported_datasets': instance.supported_datasets,
                'class_name': algorithm_class.__name__
            }
        except Exception as e:
            result[name] = {
                'name': name,
                'description': f'Error loading: {e}',
                'supported_datasets': {},
                'class_name': algorithm_class.__name__
            }

    return result


def get_algorithm_documentation(algorithm_name: str) -> str:
    """Get detailed documentation for a specific SLAM algorithm.

    Args:
        algorithm_name: Name of the algorithm

    Returns:
        Formatted documentation string
    """
    if algorithm_name not in _SLAM_ALGORITHMS:
        available = ', '.join(_SLAM_ALGORITHMS.keys()) if _SLAM_ALGORITHMS else 'none'
        return f"Algorithm '{algorithm_name}' not found. Available: {available}"

    algorithm_class = _SLAM_ALGORITHMS[algorithm_name]

    try:
        instance = algorithm_class()

        doc_lines = [
            f"Algorithm: {instance.name}",
            "=" * 40,
            f"Class: {algorithm_class.__name__}",
            ""
        ]

        # Full docstring
        if algorithm_class.__doc__:
            doc_lines.append("Description:")
            doc_lines.append("-" * 20)
            doc_lines.append(algorithm_class.__doc__.strip())
            doc_lines.append("")

        # Supported datasets
        doc_lines.append("Supported Datasets:")
        doc_lines.append("-" * 20)
        for dataset_type, modes in instance.supported_datasets.items():
            modes_str = ', '.join(modes)
            doc_lines.append(f"  {dataset_type}: {modes_str}")

        return "\n".join(doc_lines)

    except Exception as e:
        return f"Error loading algorithm '{algorithm_name}': {e}"


# Import and register algorithms here
# This happens at module import time
try:
    from .orbslam3 import ORBSLAM3Algorithm
    register_slam_algorithm('orbslam3', ORBSLAM3Algorithm)
except ImportError:
    # ORB-SLAM3 dependencies may not be available
    pass

try:
    from .s3pogs import S3POGSAlgorithm
    register_slam_algorithm('s3pogs', S3POGSAlgorithm)
except ImportError:
    # S3PO-GS dependencies may not be available
    pass

try:
    from .mast3rslam import MASt3RSLAMAlgorithm
    register_slam_algorithm('mast3rslam', MASt3RSLAMAlgorithm)
except ImportError:
    # MASt3R-SLAM dependencies may not be available
    pass

try:
    from .gigaslam import GigaSLAMAlgorithm
    register_slam_algorithm('gigaslam', GigaSLAMAlgorithm)
except ImportError:
    # GigaSLAM dependencies may not be available
    pass

try:
    from .droidslam import DROIDSLAMAlgorithm
    register_slam_algorithm('droidslam', DROIDSLAMAlgorithm)
except ImportError:
    # DROID-SLAM dependencies may not be available
    pass

try:
    from .vggtslam import VGGTSLAMAlgorithm
    register_slam_algorithm('vggtslam', VGGTSLAMAlgorithm)
except ImportError:
    # VGGT-SLAM dependencies may not be available
    pass

try:
    from .photoslam import PhotoSLAMAlgorithm
    register_slam_algorithm('photoslam', PhotoSLAMAlgorithm)
except ImportError:
    # Photo-SLAM dependencies may not be available
    pass
