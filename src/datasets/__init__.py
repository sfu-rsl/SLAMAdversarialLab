"""Dataset loaders for SLAMAdverserialLab."""

from .base import (
    CameraIntrinsics,
    Dataset,
    MockDataset
)
from .tum import TUMDataset
from .kitti import KittiDataset
from .euroc import EuRoCDataset
from .seven_scenes import SevenScenesDataset
from .factory import (
    create_dataset,
    list_datasets,
    get_dataset_info,
    discover_datasets,
    register_dataset,
    DatasetFactory,
    factory
)
from .catalog import (
    get_tum_sequence,
    list_tum_sequences,
    get_euroc_sequence,
    list_euroc_sequences,
    get_7scenes_sequence,
    list_7scenes_sequences,
    get_datasets_base_dir,
    get_tum_base_dir,
    get_kitti_base_dir,
    get_euroc_base_dir,
    get_7scenes_base_dir,
    TUM_SEQUENCES,
    EUROC_SEQUENCES,
    SEVEN_SCENES_SEQUENCES,
)
from .download import (
    download_tum_sequence,
    ensure_tum_sequence,
    download_euroc_sequence,
    ensure_euroc_sequence,
    download_7scenes_sequence,
    ensure_7scenes_sequence,
)

__all__ = [
    # Base classes
    'CameraIntrinsics',
    'Dataset',
    'MockDataset',
    # Dataset implementations
    'TUMDataset',
    'KittiDataset',
    'EuRoCDataset',
    'SevenScenesDataset',
    # Factory
    'create_dataset',
    'list_datasets',
    'get_dataset_info',
    'discover_datasets',
    'register_dataset',
    'DatasetFactory',
    'factory',
    # Catalog - TUM
    'get_tum_sequence',
    'list_tum_sequences',
    'TUM_SEQUENCES',
    # Catalog - EuRoC
    'get_euroc_sequence',
    'list_euroc_sequences',
    'EUROC_SEQUENCES',
    # Catalog - 7-Scenes
    'get_7scenes_sequence',
    'list_7scenes_sequences',
    'SEVEN_SCENES_SEQUENCES',
    # Catalog - paths
    'get_datasets_base_dir',
    'get_tum_base_dir',
    'get_kitti_base_dir',
    'get_euroc_base_dir',
    'get_7scenes_base_dir',
    # Download
    'download_tum_sequence',
    'ensure_tum_sequence',
    'download_euroc_sequence',
    'ensure_euroc_sequence',
    'download_7scenes_sequence',
    'ensure_7scenes_sequence',
]