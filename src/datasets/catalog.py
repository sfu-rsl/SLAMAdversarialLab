"""Dataset catalog with download URLs and metadata."""

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path

from ..utils import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetEntry:
    """Catalog entry for a dataset sequence."""

    name: str  # Short name (e.g., "freiburg1_desk")
    full_name: str  # Full directory name (e.g., "rgbd_dataset_freiburg1_desk")
    url: str  # Download URL
    category: str  # Category (e.g., "freiburg1", "freiburg2", "freiburg3")
    description: str = ""
    size_mb: Optional[int] = None  # Approximate download size in MB


# TUM RGB-D Dataset Catalog
# URL pattern: https://cvg.cit.tum.de/rgbd/dataset/{category}/rgbd_dataset_{sequence_name}.tgz
TUM_BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"

TUM_SEQUENCES: Dict[str, DatasetEntry] = {
    # Freiburg1 sequences (Kinect v1, handheld)
    "freiburg1_360": DatasetEntry(
        name="freiburg1_360",
        full_name="rgbd_dataset_freiburg1_360",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_360.tgz",
        category="freiburg1",
        description="360 degree rotation in an office",
        size_mb=471,
    ),
    "freiburg1_desk": DatasetEntry(
        name="freiburg1_desk",
        full_name="rgbd_dataset_freiburg1_desk",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk.tgz",
        category="freiburg1",
        description="Office desk scene",
        size_mb=244,
    ),
    "freiburg1_desk2": DatasetEntry(
        name="freiburg1_desk2",
        full_name="rgbd_dataset_freiburg1_desk2",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_desk2.tgz",
        category="freiburg1",
        description="Office desk scene (variant 2)",
        size_mb=280,
    ),
    "freiburg1_floor": DatasetEntry(
        name="freiburg1_floor",
        full_name="rgbd_dataset_freiburg1_floor",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_floor.tgz",
        category="freiburg1",
        description="Floor texture sequence",
        size_mb=403,
    ),
    "freiburg1_plant": DatasetEntry(
        name="freiburg1_plant",
        full_name="rgbd_dataset_freiburg1_plant",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
        category="freiburg1",
        description="Plant on desk",
        size_mb=230,
    ),
    "freiburg1_room": DatasetEntry(
        name="freiburg1_room",
        full_name="rgbd_dataset_freiburg1_room",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_room.tgz",
        category="freiburg1",
        description="Full room sequence",
        size_mb=494,
    ),
    "freiburg1_rpy": DatasetEntry(
        name="freiburg1_rpy",
        full_name="rgbd_dataset_freiburg1_rpy",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_rpy.tgz",
        category="freiburg1",
        description="Roll-pitch-yaw motion",
        size_mb=230,
    ),
    "freiburg1_teddy": DatasetEntry(
        name="freiburg1_teddy",
        full_name="rgbd_dataset_freiburg1_teddy",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_teddy.tgz",
        category="freiburg1",
        description="Teddy bear on desk",
        size_mb=283,
    ),
    "freiburg1_xyz": DatasetEntry(
        name="freiburg1_xyz",
        full_name="rgbd_dataset_freiburg1_xyz",
        url=f"{TUM_BASE_URL}/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
        category="freiburg1",
        description="Translation along X, Y, Z axes",
        size_mb=184,
    ),

    # Freiburg2 sequences (Kinect v1, robot platform - more accurate motion)
    "freiburg2_desk": DatasetEntry(
        name="freiburg2_desk",
        full_name="rgbd_dataset_freiburg2_desk",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_desk.tgz",
        category="freiburg2",
        description="Office desk scene (robot platform)",
        size_mb=700,
    ),
    "freiburg2_xyz": DatasetEntry(
        name="freiburg2_xyz",
        full_name="rgbd_dataset_freiburg2_xyz",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_xyz.tgz",
        category="freiburg2",
        description="Translation along X, Y, Z axes (robot platform)",
        size_mb=177,
    ),
    "freiburg2_rpy": DatasetEntry(
        name="freiburg2_rpy",
        full_name="rgbd_dataset_freiburg2_rpy",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_rpy.tgz",
        category="freiburg2",
        description="Roll-pitch-yaw motion (robot platform)",
        size_mb=320,
    ),
    "freiburg2_pioneer_360": DatasetEntry(
        name="freiburg2_pioneer_360",
        full_name="rgbd_dataset_freiburg2_pioneer_360",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_pioneer_360.tgz",
        category="freiburg2",
        description="360 degree rotation on Pioneer robot",
        size_mb=558,
    ),
    "freiburg2_pioneer_slam": DatasetEntry(
        name="freiburg2_pioneer_slam",
        full_name="rgbd_dataset_freiburg2_pioneer_slam",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_pioneer_slam.tgz",
        category="freiburg2",
        description="SLAM sequence on Pioneer robot",
        size_mb=1172,
    ),
    "freiburg2_pioneer_slam2": DatasetEntry(
        name="freiburg2_pioneer_slam2",
        full_name="rgbd_dataset_freiburg2_pioneer_slam2",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_pioneer_slam2.tgz",
        category="freiburg2",
        description="SLAM sequence 2 on Pioneer robot",
        size_mb=1116,
    ),
    "freiburg2_pioneer_slam3": DatasetEntry(
        name="freiburg2_pioneer_slam3",
        full_name="rgbd_dataset_freiburg2_pioneer_slam3",
        url=f"{TUM_BASE_URL}/freiburg2/rgbd_dataset_freiburg2_pioneer_slam3.tgz",
        category="freiburg2",
        description="SLAM sequence 3 on Pioneer robot",
        size_mb=1062,
    ),

    # Freiburg3 sequences (Kinect v1, structured environments)
    "freiburg3_long_office_household": DatasetEntry(
        name="freiburg3_long_office_household",
        full_name="rgbd_dataset_freiburg3_long_office_household",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz",
        category="freiburg3",
        description="Long office/household sequence",
        size_mb=2800,
    ),
    "freiburg3_nostructure_texture_far": DatasetEntry(
        name="freiburg3_nostructure_texture_far",
        full_name="rgbd_dataset_freiburg3_nostructure_texture_far",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_far.tgz",
        category="freiburg3",
        description="No structure, textured, far",
        size_mb=100,
    ),
    "freiburg3_nostructure_texture_near_withloop": DatasetEntry(
        name="freiburg3_nostructure_texture_near_withloop",
        full_name="rgbd_dataset_freiburg3_nostructure_texture_near_withloop",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_nostructure_texture_near_withloop.tgz",
        category="freiburg3",
        description="No structure, textured, near with loop closure",
        size_mb=141,
    ),
    "freiburg3_structure_texture_far": DatasetEntry(
        name="freiburg3_structure_texture_far",
        full_name="rgbd_dataset_freiburg3_structure_texture_far",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_structure_texture_far.tgz",
        category="freiburg3",
        description="Structure, textured, far",
        size_mb=100,
    ),
    "freiburg3_structure_texture_near": DatasetEntry(
        name="freiburg3_structure_texture_near",
        full_name="rgbd_dataset_freiburg3_structure_texture_near",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_structure_texture_near.tgz",
        category="freiburg3",
        description="Structure, textured, near",
        size_mb=66,
    ),
    "freiburg3_walking_xyz": DatasetEntry(
        name="freiburg3_walking_xyz",
        full_name="rgbd_dataset_freiburg3_walking_xyz",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_walking_xyz.tgz",
        category="freiburg3",
        description="Walking with XYZ translation (dynamic)",
        size_mb=194,
    ),
    "freiburg3_walking_halfsphere": DatasetEntry(
        name="freiburg3_walking_halfsphere",
        full_name="rgbd_dataset_freiburg3_walking_halfsphere",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_walking_halfsphere.tgz",
        category="freiburg3",
        description="Walking with halfsphere motion (dynamic)",
        size_mb=204,
    ),
    "freiburg3_sitting_xyz": DatasetEntry(
        name="freiburg3_sitting_xyz",
        full_name="rgbd_dataset_freiburg3_sitting_xyz",
        url=f"{TUM_BASE_URL}/freiburg3/rgbd_dataset_freiburg3_sitting_xyz.tgz",
        category="freiburg3",
        description="Sitting person with XYZ translation (dynamic)",
        size_mb=165,
    ),
}


def get_tum_sequence(name: str) -> Optional[DatasetEntry]:
    """
    Get TUM sequence entry by name.

    Supports various naming formats:
    - Short: "freiburg1_desk"
    - Full: "rgbd_dataset_freiburg1_desk"
    - With prefix: "tum_freiburg1_desk"

    Args:
        name: Sequence name in any supported format

    Returns:
        DatasetEntry or None if not found
    """
    # Normalize name
    name = name.lower().strip()

    # Remove common prefixes
    if name.startswith("rgbd_dataset_"):
        name = name[len("rgbd_dataset_"):]
    if name.startswith("tum_"):
        name = name[len("tum_"):]

    return TUM_SEQUENCES.get(name)


def list_tum_sequences(category: Optional[str] = None) -> list:
    """
    List available TUM sequences.

    Args:
        category: Filter by category (freiburg1, freiburg2, freiburg3)

    Returns:
        List of sequence names
    """
    if category:
        return [name for name, entry in TUM_SEQUENCES.items()
                if entry.category == category]
    return list(TUM_SEQUENCES.keys())


def get_datasets_base_dir() -> Path:
    """
    Get the base directory for datasets.

    Checks in order:
    1. SLAMADVERSERIALLAB_DATA_DIR environment variable
    2. ./datasets relative to project root

    Returns:
        Path to datasets directory
    """
    import os

    env_dir = os.environ.get("SLAMADVERSERIALLAB_DATA_DIR")
    if env_dir:
        return Path(env_dir)

    # Default to ./datasets relative to project root
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "slamadverseriallab").is_dir() or (parent / "pyproject.toml").exists():
            return parent / "datasets"

    # Fallback to current working directory
    return Path.cwd() / "datasets"


def get_tum_base_dir() -> Path:
    """Get the base directory for TUM datasets."""
    return get_datasets_base_dir() / "TUM"


def get_kitti_base_dir() -> Path:
    """Get the base directory for KITTI datasets."""
    return get_datasets_base_dir() / "kitti"


def get_euroc_base_dir() -> Path:
    """Get the base directory for EuRoC datasets."""
    return get_datasets_base_dir() / "euroc"


# EuRoC MAV Dataset Catalog
# URL pattern: http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/{category}/{sequence}/{sequence}.zip
EUROC_BASE_URL = "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset"

EUROC_SEQUENCES: Dict[str, DatasetEntry] = {
    # Vicon Room 1 sequences
    "V1_01_easy": DatasetEntry(
        name="V1_01_easy",
        full_name="V1_01_easy",
        url=f"{EUROC_BASE_URL}/vicon_room1/V1_01_easy/V1_01_easy.zip",
        category="vicon_room1",
        description="Vicon Room 1 - Easy trajectory",
        size_mb=1200,
    ),
    "V1_02_medium": DatasetEntry(
        name="V1_02_medium",
        full_name="V1_02_medium",
        url=f"{EUROC_BASE_URL}/vicon_room1/V1_02_medium/V1_02_medium.zip",
        category="vicon_room1",
        description="Vicon Room 1 - Medium difficulty",
        size_mb=900,
    ),
    "V1_03_difficult": DatasetEntry(
        name="V1_03_difficult",
        full_name="V1_03_difficult",
        url=f"{EUROC_BASE_URL}/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
        category="vicon_room1",
        description="Vicon Room 1 - Difficult trajectory",
        size_mb=900,
    ),

    # Vicon Room 2 sequences
    "V2_01_easy": DatasetEntry(
        name="V2_01_easy",
        full_name="V2_01_easy",
        url=f"{EUROC_BASE_URL}/vicon_room2/V2_01_easy/V2_01_easy.zip",
        category="vicon_room2",
        description="Vicon Room 2 - Easy trajectory",
        size_mb=1100,
    ),
    "V2_02_medium": DatasetEntry(
        name="V2_02_medium",
        full_name="V2_02_medium",
        url=f"{EUROC_BASE_URL}/vicon_room2/V2_02_medium/V2_02_medium.zip",
        category="vicon_room2",
        description="Vicon Room 2 - Medium difficulty",
        size_mb=1100,
    ),
    "V2_03_difficult": DatasetEntry(
        name="V2_03_difficult",
        full_name="V2_03_difficult",
        url=f"{EUROC_BASE_URL}/vicon_room2/V2_03_difficult/V2_03_difficult.zip",
        category="vicon_room2",
        description="Vicon Room 2 - Difficult trajectory",
        size_mb=900,
    ),

    # Machine Hall sequences
    "MH_01_easy": DatasetEntry(
        name="MH_01_easy",
        full_name="MH_01_easy",
        url=f"{EUROC_BASE_URL}/machine_hall/MH_01_easy/MH_01_easy.zip",
        category="machine_hall",
        description="Machine Hall - Easy trajectory",
        size_mb=1500,
    ),
    "MH_02_easy": DatasetEntry(
        name="MH_02_easy",
        full_name="MH_02_easy",
        url=f"{EUROC_BASE_URL}/machine_hall/MH_02_easy/MH_02_easy.zip",
        category="machine_hall",
        description="Machine Hall - Easy trajectory (variant)",
        size_mb=1200,
    ),
    "MH_03_medium": DatasetEntry(
        name="MH_03_medium",
        full_name="MH_03_medium",
        url=f"{EUROC_BASE_URL}/machine_hall/MH_03_medium/MH_03_medium.zip",
        category="machine_hall",
        description="Machine Hall - Medium difficulty",
        size_mb=1300,
    ),
    "MH_04_difficult": DatasetEntry(
        name="MH_04_difficult",
        full_name="MH_04_difficult",
        url=f"{EUROC_BASE_URL}/machine_hall/MH_04_difficult/MH_04_difficult.zip",
        category="machine_hall",
        description="Machine Hall - Difficult trajectory",
        size_mb=1100,
    ),
    "MH_05_difficult": DatasetEntry(
        name="MH_05_difficult",
        full_name="MH_05_difficult",
        url=f"{EUROC_BASE_URL}/machine_hall/MH_05_difficult/MH_05_difficult.zip",
        category="machine_hall",
        description="Machine Hall - Difficult trajectory (variant)",
        size_mb=1100,
    ),
}


def get_euroc_sequence(name: str) -> Optional[DatasetEntry]:
    """
    Get EuRoC sequence entry by name.

    Supports various naming formats:
    - Standard: "V1_01_easy", "MH_01_easy"
    - Lowercase: "v1_01_easy", "mh_01_easy"
    - No underscores: "V101easy", "MH01easy"

    Args:
        name: Sequence name in any supported format

    Returns:
        DatasetEntry or None if not found
    """
    # Try exact match first
    if name in EUROC_SEQUENCES:
        return EUROC_SEQUENCES[name]

    # Try case-insensitive match
    name_lower = name.lower()
    for seq_name, entry in EUROC_SEQUENCES.items():
        if seq_name.lower() == name_lower:
            return entry

    # Try without underscores
    name_no_underscore = name_lower.replace("_", "")
    for seq_name, entry in EUROC_SEQUENCES.items():
        if seq_name.lower().replace("_", "") == name_no_underscore:
            return entry

    return None


def list_euroc_sequences(category: Optional[str] = None) -> list:
    """
    List available EuRoC sequences.

    Args:
        category: Filter by category (vicon_room1, vicon_room2, machine_hall)

    Returns:
        List of sequence names
    """
    if category:
        return [name for name, entry in EUROC_SEQUENCES.items()
                if entry.category == category]
    return list(EUROC_SEQUENCES.keys())


# 7-Scenes RGB-D Dataset Catalog
# URL pattern: https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/{scene}.zip
SEVEN_SCENES_BASE_URL = "https://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8"

SEVEN_SCENES_SEQUENCES: Dict[str, DatasetEntry] = {
    # Chess scene
    "chess": DatasetEntry(
        name="chess",
        full_name="chess",
        url=f"{SEVEN_SCENES_BASE_URL}/chess.zip",
        category="chess",
        description="Chess board scene - 4 training, 2 test sequences",
        size_mb=250,
    ),
    # Fire scene
    "fire": DatasetEntry(
        name="fire",
        full_name="fire",
        url=f"{SEVEN_SCENES_BASE_URL}/fire.zip",
        category="fire",
        description="Fire extinguisher scene - 2 training, 2 test sequences",
        size_mb=250,
    ),
    # Heads scene
    "heads": DatasetEntry(
        name="heads",
        full_name="heads",
        url=f"{SEVEN_SCENES_BASE_URL}/heads.zip",
        category="heads",
        description="Heads/busts scene - 1 training, 1 test sequence",
        size_mb=150,
    ),
    # Office scene
    "office": DatasetEntry(
        name="office",
        full_name="office",
        url=f"{SEVEN_SCENES_BASE_URL}/office.zip",
        category="office",
        description="Office desk scene - 10 training, 2 test sequences",
        size_mb=500,
    ),
    # Pumpkin scene
    "pumpkin": DatasetEntry(
        name="pumpkin",
        full_name="pumpkin",
        url=f"{SEVEN_SCENES_BASE_URL}/pumpkin.zip",
        category="pumpkin",
        description="Pumpkin decoration scene - 4 training, 2 test sequences",
        size_mb=350,
    ),
    # Red Kitchen scene
    "redkitchen": DatasetEntry(
        name="redkitchen",
        full_name="redkitchen",
        url=f"{SEVEN_SCENES_BASE_URL}/redkitchen.zip",
        category="redkitchen",
        description="Red kitchen scene - 7 training, 5 test sequences",
        size_mb=650,
    ),
    # Stairs scene
    "stairs": DatasetEntry(
        name="stairs",
        full_name="stairs",
        url=f"{SEVEN_SCENES_BASE_URL}/stairs.zip",
        category="stairs",
        description="Staircase scene - 4 training, 2 test sequences",
        size_mb=300,
    ),
}


def get_7scenes_base_dir() -> Path:
    """Get the base directory for 7-Scenes datasets."""
    return get_datasets_base_dir() / "7scenes"


def get_7scenes_sequence(name: str) -> Optional[DatasetEntry]:
    """
    Get 7-Scenes sequence entry by name.

    Supports formats:
    - Scene only: "chess", "fire", "office"
    - Scene/sequence: "chess/seq-01", "fire/seq-02"

    Args:
        name: Sequence name (scene or scene/seq-XX)

    Returns:
        DatasetEntry or None if not found
    """
    # Normalize name
    name = name.lower().strip()

    # Extract scene name (before any /)
    scene_name = name.split('/')[0]

    return SEVEN_SCENES_SEQUENCES.get(scene_name)


def list_7scenes_sequences(scene: Optional[str] = None) -> list:
    """
    List available 7-Scenes sequences.

    Args:
        scene: Filter by scene name (chess, fire, heads, office, pumpkin, redkitchen, stairs)

    Returns:
        List of scene names
    """
    if scene:
        scene = scene.lower()
        if scene in SEVEN_SCENES_SEQUENCES:
            return [scene]
        return []
    return list(SEVEN_SCENES_SEQUENCES.keys())
