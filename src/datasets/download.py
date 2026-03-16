"""Dataset download utilities."""

import os
import tarfile
import zipfile
import urllib.request
import shutil
from pathlib import Path
from typing import Optional, Callable

from ..utils import get_logger
from .catalog import (
    get_tum_sequence,
    get_tum_base_dir,
    get_euroc_sequence,
    get_euroc_base_dir,
    get_7scenes_sequence,
    get_7scenes_base_dir,
    DatasetEntry,
)

logger = get_logger(__name__)


class DownloadProgressBar:
    """Simple progress bar for downloads."""

    def __init__(self, total: int, desc: str = "Downloading"):
        """Initialize progress bar."""
        self.total = total
        self.desc = desc
        self.downloaded = 0
        self.last_percent = -1

    def update(self, block_count: int, block_size: int, total_size: int) -> None:
        """Update progress bar (urllib reporthook signature)."""
        if total_size > 0:
            self.total = total_size

        self.downloaded = block_count * block_size
        percent = int(100 * self.downloaded / self.total) if self.total > 0 else 0

        if percent != self.last_percent:
            self.last_percent = percent
            downloaded_mb = self.downloaded / (1024 * 1024)
            total_mb = self.total / (1024 * 1024)
            print(f"\r{self.desc}: {downloaded_mb:.1f}/{total_mb:.1f} MB ({percent}%)", end="", flush=True)

    def close(self) -> None:
        """Close progress bar."""
        print()  # New line


def download_file(
    url: str,
    dest_path: Path,
    desc: str = "Downloading",
    progress: bool = True,
) -> Path:
    """
    Download a file from URL.

    Args:
        url: URL to download from
        dest_path: Destination file path
        desc: Description for progress bar
        progress: Show progress bar

    Returns:
        Path to downloaded file

    Raises:
        RuntimeError: If download fails
    """
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url}")
    logger.info(f"Destination: {dest_path}")

    try:
        if progress:
            pbar = DownloadProgressBar(total=0, desc=desc)
            urllib.request.urlretrieve(url, dest_path, reporthook=pbar.update)
            pbar.close()
        else:
            urllib.request.urlretrieve(url, dest_path)

        logger.info(f"Download complete: {dest_path}")
        return dest_path

    except Exception as e:
        # Clean up partial download
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Download failed: {e}") from e


def extract_tarball(
    archive_path: Path,
    extract_dir: Path,
    remove_archive: bool = True,
) -> Path:
    """
    Extract a tarball (.tgz, .tar.gz).

    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        remove_archive: Remove archive after extraction

    Returns:
        Path to extracted directory

    Raises:
        RuntimeError: If extraction fails
    """
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)

    logger.info(f"Extracting {archive_path}")
    logger.info(f"Extract to: {extract_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            members = tar.getmembers()
            if members:
                root_name = members[0].name.split("/")[0]
            else:
                root_name = archive_path.stem.replace(".tar", "")

            # Extract all
            tar.extractall(extract_dir)

        extracted_path = extract_dir / root_name

        if remove_archive:
            logger.debug(f"Removing archive: {archive_path}")
            archive_path.unlink()

        logger.info(f"Extraction complete: {extracted_path}")
        return extracted_path

    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}") from e


def download_tum_sequence(
    sequence_name: str,
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a TUM RGB-D sequence.

    Args:
        sequence_name: Name of the sequence (e.g., "freiburg1_desk")
        dest_dir: Destination directory (default: datasets/TUM)
        force: Force re-download even if exists

    Returns:
        Path to the downloaded sequence directory

    Raises:
        ValueError: If sequence name is unknown
        RuntimeError: If download or extraction fails
    """
    # Look up sequence in catalog
    entry = get_tum_sequence(sequence_name)
    if entry is None:
        from .catalog import list_tum_sequences
        available = list_tum_sequences()
        raise ValueError(
            f"Unknown TUM sequence '{sequence_name}'.\n"
            f"Available sequences: {', '.join(sorted(available))}"
        )

    if dest_dir is None:
        dest_dir = get_tum_base_dir()
    dest_dir = Path(dest_dir)

    sequence_dir = dest_dir / entry.full_name

    if sequence_dir.exists() and not force:
        logger.info(f"Sequence already exists: {sequence_dir}")
        return sequence_dir

    if force and sequence_dir.exists():
        logger.info(f"Force re-download, removing existing: {sequence_dir}")
        shutil.rmtree(sequence_dir)

    # Download
    archive_name = f"{entry.full_name}.tgz"
    archive_path = dest_dir / archive_name

    logger.info(f"Downloading TUM sequence: {entry.name}")
    if entry.size_mb:
        logger.info(f"Approximate size: {entry.size_mb} MB")

    download_file(
        url=entry.url,
        dest_path=archive_path,
        desc=f"Downloading {entry.name}",
    )

    # Extract
    extracted = extract_tarball(
        archive_path=archive_path,
        extract_dir=dest_dir,
        remove_archive=True,
    )

    logger.info(f"TUM sequence ready: {extracted}")
    return extracted


def ensure_tum_sequence(sequence_name: str) -> Path:
    """
    Ensure a TUM sequence is available, downloading if necessary.

    This is the main entry point for auto-download functionality.

    Args:
        sequence_name: Name of the sequence

    Returns:
        Path to the sequence directory
    """
    entry = get_tum_sequence(sequence_name)
    if entry is None:
        from .catalog import list_tum_sequences
        available = list_tum_sequences()
        raise ValueError(
            f"Unknown TUM sequence '{sequence_name}'.\n"
            f"Available sequences: {', '.join(sorted(available))}"
        )

    dest_dir = get_tum_base_dir()
    sequence_dir = dest_dir / entry.full_name

    if sequence_dir.exists():
        if (sequence_dir / "rgb").exists() or (sequence_dir / "rgb.txt").exists():
            logger.debug(f"TUM sequence exists: {sequence_dir}")
            return sequence_dir
        else:
            logger.warning(f"Sequence directory exists but appears incomplete: {sequence_dir}")
            # Could add option to re-download here

    # Download
    return download_tum_sequence(sequence_name, dest_dir)


def extract_zipfile(
    archive_path: Path,
    extract_dir: Path,
    remove_archive: bool = True,
) -> Path:
    """
    Extract a ZIP archive.

    Args:
        archive_path: Path to archive file
        extract_dir: Directory to extract to
        remove_archive: Remove archive after extraction

    Returns:
        Path to extracted directory

    Raises:
        RuntimeError: If extraction fails
    """
    archive_path = Path(archive_path)
    extract_dir = Path(extract_dir)

    logger.info(f"Extracting {archive_path}")
    logger.info(f"Extract to: {extract_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            namelist = zip_ref.namelist()
            if namelist:
                root_name = namelist[0].split("/")[0]
            else:
                root_name = archive_path.stem

            # Extract files one by one to handle corrupt files gracefully
            for member in zip_ref.namelist():
                try:
                    zip_ref.extract(member, extract_dir)
                except zipfile.BadZipFile as e:
                    # Skip corrupt files (e.g., Thumbs.db with bad CRC)
                    logger.warning(f"Skipping corrupt file in archive: {member} ({e})")
                except Exception as e:
                    logger.warning(f"Failed to extract {member}: {e}")

        extracted_path = extract_dir / root_name

        if remove_archive:
            logger.debug(f"Removing archive: {archive_path}")
            archive_path.unlink()

        logger.info(f"Extraction complete: {extracted_path}")
        return extracted_path

    except Exception as e:
        raise RuntimeError(f"Extraction failed: {e}") from e


def download_euroc_sequence(
    sequence_name: str,
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download an EuRoC MAV sequence.

    Args:
        sequence_name: Name of the sequence (e.g., "V1_01_easy", "MH_01_easy")
        dest_dir: Destination directory (default: datasets/euroc)
        force: Force re-download even if exists

    Returns:
        Path to the downloaded sequence directory

    Raises:
        ValueError: If sequence name is unknown
        RuntimeError: If download or extraction fails
    """
    # Look up sequence in catalog
    entry = get_euroc_sequence(sequence_name)
    if entry is None:
        from .catalog import list_euroc_sequences
        available = list_euroc_sequences()
        raise ValueError(
            f"Unknown EuRoC sequence '{sequence_name}'.\n"
            f"Available sequences: {', '.join(sorted(available))}"
        )

    if dest_dir is None:
        dest_dir = get_euroc_base_dir()
    dest_dir = Path(dest_dir)

    sequence_dir = dest_dir / entry.full_name

    if sequence_dir.exists() and not force:
        logger.info(f"Sequence already exists: {sequence_dir}")
        return sequence_dir

    if force and sequence_dir.exists():
        logger.info(f"Force re-download, removing existing: {sequence_dir}")
        shutil.rmtree(sequence_dir)

    # Download
    archive_name = f"{entry.full_name}.zip"
    archive_path = dest_dir / archive_name

    logger.info(f"Downloading EuRoC sequence: {entry.name}")
    if entry.size_mb:
        logger.info(f"Approximate size: {entry.size_mb} MB")

    download_file(
        url=entry.url,
        dest_path=archive_path,
        desc=f"Downloading {entry.name}",
    )

    # Extract to a temp location first, then rename
    # EuRoC ZIP files contain 'mav0/' as root, not the sequence name
    temp_extract_dir = dest_dir / f"_temp_{entry.full_name}"
    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    extract_zipfile(
        archive_path=archive_path,
        extract_dir=temp_extract_dir,
        remove_archive=True,
    )

    # The ZIP extracts to temp_dir/mav0/, we need to move it to sequence_dir/
    extracted_mav0 = temp_extract_dir / "mav0"
    if extracted_mav0.exists():
        sequence_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_mav0), str(sequence_dir / "mav0"))
        # Clean up temp dir and __MACOSX if present
        shutil.rmtree(temp_extract_dir)
    else:
        # Fallback: rename temp dir to sequence dir
        shutil.move(str(temp_extract_dir), str(sequence_dir))

    logger.info(f"EuRoC sequence ready: {sequence_dir}")
    return sequence_dir


def ensure_euroc_sequence(sequence_name: str) -> Path:
    """
    Ensure an EuRoC sequence is available, downloading if necessary.

    This is the main entry point for auto-download functionality.

    Args:
        sequence_name: Name of the sequence

    Returns:
        Path to the sequence directory
    """
    entry = get_euroc_sequence(sequence_name)
    if entry is None:
        from .catalog import list_euroc_sequences
        available = list_euroc_sequences()
        raise ValueError(
            f"Unknown EuRoC sequence '{sequence_name}'.\n"
            f"Available sequences: {', '.join(sorted(available))}"
        )

    dest_dir = get_euroc_base_dir()
    sequence_dir = dest_dir / entry.full_name

    if sequence_dir.exists():
        cam0_data = sequence_dir / "mav0" / "cam0" / "data"
        if cam0_data.exists():
            logger.debug(f"EuRoC sequence exists: {sequence_dir}")
            return sequence_dir
        else:
            logger.warning(f"Sequence directory exists but appears incomplete: {sequence_dir}")

    # Download
    return download_euroc_sequence(sequence_name, dest_dir)


def download_7scenes_sequence(
    sequence_name: str,
    dest_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Download a 7-Scenes sequence.

    Args:
        sequence_name: Scene name or scene/seq-XX (e.g., "chess", "chess/seq-01")
        dest_dir: Destination directory (default: datasets/7scenes)
        force: Force re-download even if exists

    Returns:
        Path to the downloaded sequence directory

    Raises:
        ValueError: If sequence name is unknown
        RuntimeError: If download or extraction fails
    """
    # Parse sequence name
    parts = sequence_name.lower().split('/')
    scene_name = parts[0]
    seq_name = parts[1] if len(parts) > 1 else None

    # Look up scene in catalog
    entry = get_7scenes_sequence(scene_name)
    if entry is None:
        from .catalog import list_7scenes_sequences
        available = list_7scenes_sequences()
        raise ValueError(
            f"Unknown 7-Scenes scene '{scene_name}'.\n"
            f"Available scenes: {', '.join(sorted(available))}"
        )

    if dest_dir is None:
        dest_dir = get_7scenes_base_dir()
    dest_dir = Path(dest_dir)

    scene_dir = dest_dir / scene_name
    if seq_name:
        sequence_dir = scene_dir / seq_name
    else:
        sequence_dir = scene_dir

    if sequence_dir.exists() and not force:
        logger.info(f"Sequence already exists: {sequence_dir}")
        return sequence_dir

    # Need to download the whole scene even if only one sequence is requested
    if scene_dir.exists() and not force:
        # Scene exists, check if specific sequence exists
        if seq_name and not (scene_dir / seq_name).exists():
            logger.warning(f"Scene exists but sequence {seq_name} not found")
        return sequence_dir

    if force and scene_dir.exists():
        logger.info(f"Force re-download, removing existing: {scene_dir}")
        shutil.rmtree(scene_dir)

    # Download
    archive_name = f"{scene_name}.zip"
    archive_path = dest_dir / archive_name

    logger.info(f"Downloading 7-Scenes scene: {scene_name}")
    if entry.size_mb:
        logger.info(f"Approximate size: {entry.size_mb} MB")

    download_file(
        url=entry.url,
        dest_path=archive_path,
        desc=f"Downloading {scene_name}",
    )

    # Extract main scene zip
    extract_zipfile(
        archive_path=archive_path,
        extract_dir=dest_dir,
        remove_archive=True,
    )

    # 7-Scenes has nested zips - each sequence is a separate zip file inside
    # Extract all seq-XX.zip files inside the scene directory
    for seq_zip in sorted(scene_dir.glob("seq-*.zip")):
        logger.info(f"Extracting sequence: {seq_zip.name}")
        extract_zipfile(
            archive_path=seq_zip,
            extract_dir=scene_dir,
            remove_archive=True,
        )

    logger.info(f"7-Scenes scene ready: {scene_dir}")
    return sequence_dir


def ensure_7scenes_sequence(sequence_name: str) -> Path:
    """
    Ensure a 7-Scenes sequence is available, downloading if necessary.

    This is the main entry point for auto-download functionality.

    Args:
        sequence_name: Scene name or scene/seq-XX

    Returns:
        Path to the sequence directory
    """
    # Parse sequence name
    parts = sequence_name.lower().split('/')
    scene_name = parts[0]
    seq_name = parts[1] if len(parts) > 1 else None

    entry = get_7scenes_sequence(scene_name)
    if entry is None:
        from .catalog import list_7scenes_sequences
        available = list_7scenes_sequences()
        raise ValueError(
            f"Unknown 7-Scenes scene '{scene_name}'.\n"
            f"Available scenes: {', '.join(sorted(available))}"
        )

    dest_dir = get_7scenes_base_dir()
    scene_dir = dest_dir / scene_name

    if seq_name:
        sequence_dir = scene_dir / seq_name
    else:
        sequence_dir = scene_dir

    if sequence_dir.exists():
        if seq_name:
            if list(sequence_dir.glob("frame-*.color.png")):
                logger.debug(f"7-Scenes sequence exists: {sequence_dir}")
                return sequence_dir
        else:
            if list(scene_dir.glob("seq-*")):
                logger.debug(f"7-Scenes scene exists: {scene_dir}")
                return sequence_dir

        logger.warning(f"Sequence directory exists but appears incomplete: {sequence_dir}")

    # Download
    return download_7scenes_sequence(sequence_name, dest_dir)
