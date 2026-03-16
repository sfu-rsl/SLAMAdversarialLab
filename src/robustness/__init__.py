"""Robustness-boundary shared contracts and helpers."""

from .param_spec import (
    BoundaryParamSpec,
    apply_canonicalize,
    format_trial_value,
    is_interval_small_enough,
    is_param_active,
    midpoint,
    parse_domain_value,
)

__all__ = [
    "BoundaryParamSpec",
    "apply_canonicalize",
    "format_trial_value",
    "is_interval_small_enough",
    "is_param_active",
    "midpoint",
    "parse_domain_value",
]

