"""Dataset factory and registry."""

import importlib
from dataclasses import replace
from typing import Dict, Type, Optional, Any, List
from pathlib import Path

from .base import Dataset
from ..config import DatasetConfig
from ..utils import get_logger

logger = get_logger(__name__)


class DatasetRegistry:
    """Registry for available dataset types."""

    def __init__(self):
        """Initialize the dataset registry."""
        self._datasets: Dict[str, Type[Dataset]] = {}
        self._aliases: Dict[str, str] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        dataset_class: Type[Dataset],
        aliases: Optional[List[str]] = None,
        description: Optional[str] = None,
        **metadata
    ) -> None:
        """
        Register a dataset type.

        Args:
            name: Primary name for the dataset type
            dataset_class: Dataset class to register
            aliases: Alternative names for the dataset
            description: Human-readable description
            **metadata: Additional metadata about the dataset
        """
        if name in self._datasets:
            logger.warning(f"Overwriting existing dataset type: {name}")

        self._datasets[name.lower()] = dataset_class

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias.lower()] = name.lower()

        # Store metadata
        self._metadata[name.lower()] = {
            'class': dataset_class,
            'description': description or f"{name} dataset",
            **metadata
        }

        logger.debug(f"Registered dataset type: {name} ({dataset_class.__name__})")

    def get(self, name: str) -> Optional[Type[Dataset]]:
        """
        Get a dataset class by name.

        Args:
            name: Dataset name or alias

        Returns:
            Dataset class or None if not found
        """
        name_lower = name.lower()

        if name_lower in self._aliases:
            name_lower = self._aliases[name_lower]

        return self._datasets.get(name_lower)

    def list(self) -> List[str]:
        """
        List all registered dataset types.

        Returns:
            List of dataset names
        """
        return list(self._datasets.keys())

    def get_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a dataset type.

        Args:
            name: Dataset name

        Returns:
            Metadata dictionary or None
        """
        name_lower = name.lower()
        if name_lower in self._aliases:
            name_lower = self._aliases[name_lower]
        return self._metadata.get(name_lower)

    def __contains__(self, name: str) -> bool:
        """Check if a dataset type is registered."""
        name_lower = name.lower()
        return name_lower in self._datasets or name_lower in self._aliases

    def __repr__(self) -> str:
        """String representation."""
        return f"DatasetRegistry({', '.join(self.list())})"


# Global registry instance
_registry = DatasetRegistry()


def register_dataset(
    name: str,
    aliases: Optional[List[str]] = None,
    description: Optional[str] = None,
    **metadata
):
    """Decorator for registering dataset classes.

    Args:
        name: Primary name for the dataset
        aliases: Alternative names
        description: Human-readable description
        **metadata: Additional metadata
    """
    def decorator(cls: Type[Dataset]) -> Type[Dataset]:
        _registry.register(name, cls, aliases, description, **metadata)
        return cls
    return decorator


def create_dataset(config: DatasetConfig) -> Dataset:
    """
    Create a dataset instance from configuration.

    Args:
        config: Dataset configuration

    Returns:
        Initialized dataset instance

    Raises:
        ValueError: If dataset type is unknown
        RuntimeError: If dataset creation fails
    """
    dataset_type = config.type.lower()

    if dataset_type not in _registry:
        available = _registry.list()
        raise ValueError(
            f"Unknown dataset type '{dataset_type}'. "
            f"Available types: {', '.join(available)}"
        )

    dataset_class = _registry.get(dataset_type)

    effective_config = config

    if not config.path and config.sequence:
        resolved_path = dataset_class.resolve_path(config)
        logger.info(f"Resolved sequence '{config.sequence}' to path: {resolved_path}")
        effective_config = replace(config, path=resolved_path)

    logger.info(f"Creating dataset: {dataset_type}")
    logger.debug(f"Dataset class: {dataset_class.__name__}")
    logger.debug(f"Dataset path: {effective_config.path}")

    try:
        dataset = dataset_class(effective_config)

        return dataset

    except Exception as e:
        logger.error(f"Failed to create dataset '{dataset_type}': {e}")
        raise RuntimeError(f"Dataset creation failed: {e}") from e


def list_datasets() -> List[str]:
    """
    List all available dataset types.

    Returns:
        List of dataset names
    """
    return _registry.list()


def get_dataset_info(name: str) -> Optional[Dict[str, Any]]:
    """
    Get information about a dataset type.

    Args:
        name: Dataset name

    Returns:
        Dictionary with dataset information or None
    """
    metadata = _registry.get_metadata(name)
    if metadata:
        return {
            'name': name,
            'class': metadata['class'].__name__,
            'description': metadata.get('description'),
            'module': metadata['class'].__module__,
            **{k: v for k, v in metadata.items()
               if k not in ['class', 'description']}
        }
    return None


def auto_register_datasets():
    """
    Automatically register built-in dataset types.

    This function imports and registers all known dataset types.
    It's called automatically when the module is imported.
    """
    # Register MockDataset (always available)
    from .base import MockDataset
    _registry.register(
        "mock",
        MockDataset,
        aliases=["test", "dummy"],
        description="Mock dataset for testing",
        synthetic=True
    )

    # Try to register TUM dataset
    try:
        from .tum import TUMDataset
        _registry.register(
            "tum",
            TUMDataset,
            aliases=["tum-rgbd", "tumrgbd"],
            description="TUM RGB-D dataset for indoor SLAM",
            has_depth=True,
            has_ground_truth=True
        )
    except ImportError as e:
        logger.debug(f"TUM dataset not available: {e}")

    # Try to register KITTI dataset
    try:
        from .kitti import KittiDataset
        _registry.register(
            "kitti",
            KittiDataset,
            aliases=["kitti-odometry"],
            description="KITTI dataset for outdoor SLAM",
            has_ground_truth=True,
            outdoor=True
        )
    except ImportError as e:
        logger.debug(f"KITTI dataset not available: {e}")

    # Try to register EuRoC dataset
    try:
        from .euroc import EuRoCDataset
        _registry.register(
            "euroc",
            EuRoCDataset,
            aliases=["euroc-mav", "eurocmav"],
            description="EuRoC MAV dataset for indoor/industrial SLAM",
            has_depth=False,
            has_ground_truth=True,
            has_imu=True,
            stereo=True
        )
    except ImportError as e:
        logger.debug(f"EuRoC dataset not available: {e}")

    # Try to register 7-Scenes dataset
    try:
        from .seven_scenes import SevenScenesDataset
        _registry.register(
            "7scenes",
            SevenScenesDataset,
            aliases=["seven-scenes", "sevenscenes", "7-scenes"],
            description="Microsoft 7-Scenes RGB-D dataset for indoor relocalization",
            has_depth=True,
            has_ground_truth=True,
            indoor=True
        )
    except ImportError as e:
        logger.debug(f"7-Scenes dataset not available: {e}")

    logger.debug(f"Auto-registered datasets: {_registry.list()}")


# Auto-register on import
auto_register_datasets()


class DatasetFactory:
    """Create datasets with optional caching and validation."""

    def __init__(self):
        """Initialize the factory."""
        self._cache: Dict[str, Dataset] = {}
        self._registry = _registry

    def create(
        self,
        config: DatasetConfig,
        use_cache: bool = False,
        validate: bool = True
    ) -> Dataset:
        """
        Create a dataset with optional caching and validation.

        Args:
            config: Dataset configuration
            use_cache: Whether to use cached datasets
            validate: Whether to validate the dataset after creation

        Returns:
            Dataset instance
        """
        cache_key = f"{config.type}_{config.path}"

        if use_cache and cache_key in self._cache:
            logger.info(f"Using cached dataset: {cache_key}")
            return self._cache[cache_key]

        dataset = create_dataset(config)

        if validate:
            self._validate_dataset(dataset, config)

        if use_cache:
            self._cache[cache_key] = dataset
            logger.debug(f"Cached dataset: {cache_key}")

        return dataset

    def _validate_dataset(self, dataset: Dataset, config: DatasetConfig):
        """
        Validate a dataset instance.

        Args:
            dataset: Dataset to validate
            config: Original configuration

        Raises:
            ValueError: If validation fails
        """
        if len(dataset) == 0:
            raise ValueError(f"Dataset is empty: {config.path}")

        if config.type != "mock":
            path = Path(config.path)
            if not path.exists():
                logger.warning(f"Dataset path does not exist: {path}")

        # Try to load first frame
        try:
            first_frame = dataset[0]
            if first_frame is None or first_frame.get('image') is None:
                raise ValueError("First frame is invalid")
        except Exception as e:
            raise ValueError(f"Cannot load first frame: {e}")

        logger.debug("Dataset validation passed")

    def clear_cache(self):
        """Clear the dataset cache."""
        self._cache.clear()
        logger.debug("Dataset cache cleared")

    def list_available(self) -> List[Dict[str, Any]]:
        """
        List all available datasets with their metadata.

        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        for name in self._registry.list():
            info = get_dataset_info(name)
            if info:
                datasets.append(info)
        return datasets

    def get_dataset_class(self, name: str) -> Optional[Type[Dataset]]:
        """
        Get the dataset class for a given type.

        Args:
            name: Dataset type name

        Returns:
            Dataset class or None
        """
        return self._registry.get(name)


# Global factory instance
factory = DatasetFactory()


def discover_datasets(directory: Path) -> List[Dict[str, Any]]:
    """
    Discover datasets in a directory.

    This function scans a directory for known dataset formats
    and returns information about found datasets.

    Args:
        directory: Directory to scan

    Returns:
        List of discovered dataset information
    """
    discovered = []
    directory = Path(directory)

    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return discovered

    if (directory / "rgb.txt").exists() and (directory / "depth.txt").exists():
        discovered.append({
            'type': 'tum',
            'path': str(directory),
            'name': directory.name,
            'format': 'TUM RGB-D'
        })

    if (directory / "calib.txt").exists():
        discovered.append({
            'type': 'kitti',
            'path': str(directory),
            'name': directory.name,
            'format': 'KITTI'
        })

    # Recursively check subdirectories (limited depth)
    for subdir in directory.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            discovered.extend(discover_datasets(subdir))

    return discovered
