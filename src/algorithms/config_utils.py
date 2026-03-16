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
