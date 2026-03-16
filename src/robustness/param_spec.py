"""Boundary parameter contract and domain helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Literal

BoundaryDomain = Literal["continuous", "integer", "bitrate_string"]


@dataclass(frozen=True)
class BoundaryParamSpec:
    """Specification for one module parameter in robustness-boundary search."""

    domain: BoundaryDomain
    active_if: Callable[[dict[str, Any]], bool] | None = None
    canonicalize: Callable[[Any], Any] | None = None


def is_param_active(spec: BoundaryParamSpec, params: dict[str, Any]) -> bool:
    """Return True when the parameter is active under current module params."""
    if spec.active_if is None:
        return True
    return bool(spec.active_if(params))


def apply_canonicalize(spec: BoundaryParamSpec, value: Any) -> Any:
    """Apply optional parameter canonicalization hook."""
    if spec.canonicalize is None:
        return value
    return spec.canonicalize(value)


def parse_domain_value(spec: BoundaryParamSpec, raw_value: Any) -> float | int:
    """Parse a raw config/search value into numeric search space."""
    if spec.domain == "continuous":
        return _parse_continuous(raw_value)
    if spec.domain == "integer":
        return _parse_integer(raw_value)
    if spec.domain == "bitrate_string":
        return _parse_bitrate_value(raw_value)
    raise ValueError(f"Unsupported boundary domain '{spec.domain}'")


def format_trial_value(spec: BoundaryParamSpec, sampled_value: Any) -> Any:
    """Format sampled search-space value to module parameter value."""
    if spec.domain == "continuous":
        return float(_parse_continuous(sampled_value))
    if spec.domain == "integer":
        return _parse_integer(sampled_value)
    if spec.domain == "bitrate_string":
        return _format_bitrate_string(_parse_bitrate_value(sampled_value))
    raise ValueError(f"Unsupported boundary domain '{spec.domain}'")


def midpoint(spec: BoundaryParamSpec, pass_bound: Any, fail_bound: Any) -> float | int:
    """Return the midpoint for a domain-aware bisection step."""
    if spec.domain == "continuous":
        p = _parse_continuous(pass_bound)
        f = _parse_continuous(fail_bound)
        return (p + f) / 2.0

    if spec.domain == "integer":
        p = _parse_integer(pass_bound)
        f = _parse_integer(fail_bound)
        return (p + f) // 2

    if spec.domain == "bitrate_string":
        p = _parse_bitrate_value(pass_bound)
        f = _parse_bitrate_value(fail_bound)
        return (p + f) // 2

    raise ValueError(f"Unsupported boundary domain '{spec.domain}'")


def is_interval_small_enough(
    spec: BoundaryParamSpec,
    pass_bound: Any,
    fail_bound: Any,
    tolerance: Any,
) -> bool:
    """Return True when boundary interval has reached stopping precision."""
    if spec.domain == "continuous":
        tol = _parse_continuous(tolerance)
        if tol <= 0:
            raise ValueError(f"Tolerance must be positive for continuous domain, got {tol}")
        p = _parse_continuous(pass_bound)
        f = _parse_continuous(fail_bound)
        return abs(f - p) <= tol

    if spec.domain == "integer":
        tol = _parse_integer_tolerance(tolerance)
        p = _parse_integer(pass_bound)
        f = _parse_integer(fail_bound)
        return abs(f - p) <= tol

    if spec.domain == "bitrate_string":
        tol = _parse_bitrate_tolerance(tolerance)
        p = _parse_bitrate_value(pass_bound)
        f = _parse_bitrate_value(fail_bound)
        return abs(f - p) <= tol

    raise ValueError(f"Unsupported boundary domain '{spec.domain}'")


def _ensure_not_bool(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")


def _parse_continuous(raw_value: Any) -> float:
    _ensure_not_bool(raw_value, "Value")
    if isinstance(raw_value, (int, float)):
        if not math.isfinite(float(raw_value)):
            raise ValueError(f"Value must be finite, got {raw_value}")
        return float(raw_value)
    if isinstance(raw_value, str):
        try:
            value = float(raw_value.strip())
        except ValueError as exc:
            raise ValueError(f"Expected numeric value, got '{raw_value}'") from exc
        if not math.isfinite(value):
            raise ValueError(f"Value must be finite, got {raw_value}")
        return value
    raise ValueError(f"Expected numeric value, got {type(raw_value).__name__}")


def _parse_integer(raw_value: Any) -> int:
    _ensure_not_bool(raw_value, "Value")
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if not math.isfinite(raw_value):
            raise ValueError(f"Value must be finite, got {raw_value}")
        if not raw_value.is_integer():
            raise ValueError(f"Expected integer value, got {raw_value}")
        return int(raw_value)
    if isinstance(raw_value, str):
        text = raw_value.strip()
        try:
            return int(text)
        except ValueError:
            try:
                value = float(text)
            except ValueError as exc:
                raise ValueError(f"Expected integer value, got '{raw_value}'") from exc
            if not math.isfinite(value) or not value.is_integer():
                raise ValueError(f"Expected integer value, got '{raw_value}'")
            return int(value)
    raise ValueError(f"Expected integer value, got {type(raw_value).__name__}")


def _parse_bitrate_value(raw_value: Any) -> int:
    _ensure_not_bool(raw_value, "Bitrate")
    if isinstance(raw_value, int):
        if raw_value <= 0:
            raise ValueError(f"Bitrate must be positive, got {raw_value}")
        return raw_value
    if isinstance(raw_value, float):
        if not math.isfinite(raw_value) or raw_value <= 0:
            raise ValueError(f"Bitrate must be positive finite value, got {raw_value}")
        if not raw_value.is_integer():
            raise ValueError(f"Bitrate in bps must be integer, got {raw_value}")
        return int(raw_value)
    if isinstance(raw_value, str):
        return _parse_bitrate_string(raw_value)
    raise ValueError(f"Expected bitrate value, got {type(raw_value).__name__}")


def _parse_bitrate_string(raw_value: str) -> int:
    text = raw_value.strip()
    if not text:
        raise ValueError("Bitrate string cannot be empty")

    unit = text[-1]
    if unit in {"k", "K", "m", "M", "g", "G"}:
        number_text = text[:-1].strip()
        if not number_text:
            raise ValueError(f"Invalid bitrate string '{raw_value}'")
        try:
            number = float(number_text)
        except ValueError as exc:
            raise ValueError(f"Invalid bitrate value '{raw_value}'") from exc
        multipliers = {
            "k": 1_000,
            "K": 1_000,
            "m": 1_000_000,
            "M": 1_000_000,
            "g": 1_000_000_000,
            "G": 1_000_000_000,
        }
        bps = int(round(number * multipliers[unit]))
    else:
        try:
            number = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid bitrate value '{raw_value}'") from exc
        if not number.is_integer():
            raise ValueError(f"Bitrate in bps must be integer, got '{raw_value}'")
        bps = int(number)

    if bps <= 0:
        raise ValueError(f"Bitrate must be positive, got '{raw_value}'")
    return bps


def _format_bitrate_string(bps: int) -> str:
    if bps <= 0:
        raise ValueError(f"Bitrate must be positive, got {bps}")
    if bps % 1_000_000 == 0:
        return f"{bps // 1_000_000}M"
    if bps % 1_000 == 0:
        return f"{bps // 1_000}k"
    return str(bps)


def _parse_integer_tolerance(tolerance: Any) -> int:
    tol = _parse_continuous(tolerance)
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    return max(1, int(math.ceil(tol)))


def _parse_bitrate_tolerance(tolerance: Any) -> int:
    if isinstance(tolerance, str):
        return _parse_bitrate_string(tolerance)
    tol = _parse_continuous(tolerance)
    if tol <= 0:
        raise ValueError(f"Tolerance must be positive, got {tol}")
    return max(1, int(math.ceil(tol)))

