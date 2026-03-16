"""Path utilities for organizing datasets and outputs."""

from pathlib import Path
from typing import Optional

# Global temp directory root - set by pipeline or evaluation before running
_TEMP_DIR_ROOT: Optional[Path] = None


def set_temp_dir_root(root: Path) -> None:
    """Set the root directory for temporary files.

    This should be called at the start of pipeline/evaluation to set
    where temp files are created (e.g., results/experiment_name/.tmp/).

    Args:
        root: Root directory for temp files
    """
    global _TEMP_DIR_ROOT
    _TEMP_DIR_ROOT = Path(root)
    _TEMP_DIR_ROOT.mkdir(parents=True, exist_ok=True)


def get_temp_dir_root() -> Path:
    """Get the root directory for temporary files.

    Returns:
        Path to temp directory root. Falls back to project .tmp/ if not set.
    """
    global _TEMP_DIR_ROOT
    if _TEMP_DIR_ROOT is None:
        # Fallback to project root .tmp/
        project_root = Path(__file__).parent.parent.parent
        _TEMP_DIR_ROOT = project_root / ".tmp"
        _TEMP_DIR_ROOT.mkdir(parents=True, exist_ok=True)
    return _TEMP_DIR_ROOT


def create_temp_dir(prefix: str = "tmp_") -> Path:
    """Create a temporary directory within the project temp root.

    This replaces tempfile.mkdtemp() to keep temp files within the project
    structure instead of /tmp.

    Args:
        prefix: Prefix for the temp directory name

    Returns:
        Path to the created temp directory
    """
    import uuid
    temp_root = get_temp_dir_root()
    temp_dir = temp_root / f"{prefix}{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def cleanup_temp_root() -> None:
    """Clean up all temporary directories in the temp root.

    Call this at the end of pipeline/evaluation to remove all temp files.
    """
    import shutil
    global _TEMP_DIR_ROOT
    if _TEMP_DIR_ROOT and _TEMP_DIR_ROOT.exists():
        shutil.rmtree(_TEMP_DIR_ROOT, ignore_errors=True)
    _TEMP_DIR_ROOT = None
