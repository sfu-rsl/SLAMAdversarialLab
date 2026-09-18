"""Configuration utilities for SLAM algorithm wrappers."""

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config_with_inherit(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file with inheritance support.

    Handles 'inherit_from' keys that specify a base config to inherit from.

    Args:
        config_path: Path to the config file

    Returns:
        Merged configuration dictionary
    """
    with open(config_path, "r") as f:
        cfg_special = yaml.safe_load(f) or {}

    inherit_from = cfg_special.get("inherit_from")

    if inherit_from is not None:
        inherit_path = Path(inherit_from)
        if not inherit_path.is_absolute():
            # Try relative to config file first
            inherit_path = config_path.parent / inherit_from
            if not inherit_path.exists():
                # Try relative to repo root (common for OpenGS-SLAM configs)
                # Go up until we find the path
                repo_root = config_path.parent
                while repo_root.parent != repo_root:
                    candidate = repo_root / inherit_from
                    if candidate.exists():
                        inherit_path = candidate
                        break
                    repo_root = repo_root.parent

        cfg = load_config_with_inherit(inherit_path)
    else:
        cfg = {}

    # Remove inherit_from from special config before merging
    cfg_special.pop("inherit_from", None)

    # Merge configs recursively
    _update_recursive(cfg, cfg_special)

    return cfg


def _update_recursive(dict1: Dict, dict2: Dict) -> None:
    """Update dict1 with values from dict2 recursively.

    Args:
        dict1: Base dictionary to update (modified in place)
        dict2: Dictionary with values to merge in
    """
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = {}
        if isinstance(v, dict) and isinstance(dict1.get(k), dict):
            _update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def tee_console_output(main_cmd: str, container_output_path: str) -> str:
    """Wrap a container shell command so its console output is persisted per run.

    Six wrappers used to run their SLAM bare, so the console never reached the
    run directory and only survived in the per-campaign log, where every repeat
    of a cell shares one file. That makes the standing rule -- a health verdict
    read from THIS run's log before its ATE is scored -- unsatisfiable, which is
    how it was found (preflight P18).

    Both streams are merged deliberately: several of these systems report
    tracking loss and reset diagnostics on stderr, so a log carrying only stdout
    would exist, pass an existence check, and still be blind to the failure
    signatures it is read for.

    Requires ``bash`` (not ``sh``) for PIPESTATUS, which every caller already
    uses. The SLAM's own exit status is preserved through the pipe rather than
    tee's, so a crash still reports as a crash.
    """
    return (
        # Unbuffered: Python block-buffers stdout when it is a pipe rather than
        # a TTY, so a SLAM that is force-stopped (the deadline harness and the
        # stop-on-line workarounds both kill their container) loses whatever is
        # still in the buffer. DPVO produced a 0-byte log for exactly this
        # reason while noisier systems happened to flush. Line-buffering costs
        # nothing here and makes the capture survive a kill.
        # The braces are load-bearing. Without grouping, `A && B && C 2>&1 | tee`
        # parses as `A && B && (C 2>&1 | tee)`, so the pipe binds ONLY to the
        # last command in the chain. DPVO's command ends in a silent `cp`, which
        # is why its log came out 0 bytes while systems whose command ends in
        # the python call happened to capture fine.
        "set +e; export PYTHONUNBUFFERED=1; "
        f"{{ {main_cmd} ; }} 2>&1 | tee {container_output_path}/slam_output.log; "
        "exit ${PIPESTATUS[0]}"
    )
