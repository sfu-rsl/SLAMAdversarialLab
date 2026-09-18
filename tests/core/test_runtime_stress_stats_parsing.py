"""Unit tests for runtime-stress stats-payload parsing.

The sampler consumes ``podman stats --format json`` and
``docker stats --format json``, which emit different shapes for the
same fields. These tests pin the dual-shape handling so we don't lose
CPU% / RSS / IO observability when switching runtimes.

These parsers are pure functions in ``runtime_stress.telemetry``; they need
no orchestrator, no request and no container.
"""

from __future__ import annotations

from slamadversariallab.runtime_stress.telemetry import (
    parse_block_io,
    parse_byte_value,
    parse_cpu_percent,
    parse_int_value,
    parse_memory_usage,
    parse_percent_value,
)


# --- Memory usage ---


def test_parse_memory_usage_podman_int_shape() -> None:
    payload = {"MemUsage": 364544, "MemLimit": 67135684608}
    rss, limit = parse_memory_usage(payload)
    assert rss == 364544
    assert limit == 67135684608


def test_parse_memory_usage_podman_int_with_no_limit() -> None:
    payload = {"MemUsage": 1000, "MemLimit": 0}
    rss, limit = parse_memory_usage(payload)
    assert rss == 1000
    assert limit == 0


def test_parse_memory_usage_docker_string_shape() -> None:
    payload = {"MemUsage": "56.4MiB / 15.5GiB"}
    rss, limit = parse_memory_usage(payload)
    assert rss is not None and rss > 56 * 1024 * 1024 * 0.9
    assert limit is not None and limit > 15 * 1024 ** 3 * 0.9


def test_parse_memory_usage_returns_none_on_unknown_shape() -> None:
    assert parse_memory_usage({}) == (None, None)
    assert parse_memory_usage({"MemUsage": None}) == (None, None)
    assert parse_memory_usage({"MemUsage": "garbage"}) == (None, None)


def test_parse_memory_usage_rejects_bool_disguised_as_int() -> None:
    # bool subclasses int in Python; make sure we don't pick up True/False.
    rss, limit = parse_memory_usage({"MemUsage": True, "MemLimit": False})
    assert rss is None and limit is None


# --- CPU percent ---


def test_parse_cpu_percent_podman_float_shape() -> None:
    assert parse_cpu_percent({"CPU": 12.34}) == 12.34
    assert parse_cpu_percent({"AvgCPU": 7.5}) == 7.5
    # CPU takes precedence over AvgCPU when both present.
    assert parse_cpu_percent({"CPU": 1.0, "AvgCPU": 99.0}) == 1.0


def test_parse_cpu_percent_podman_int_zero() -> None:
    assert parse_cpu_percent({"CPU": 0}) == 0.0


def test_parse_cpu_percent_docker_string_shape() -> None:
    assert parse_cpu_percent({"CPUPerc": "12.34%"}) == 12.34
    assert parse_cpu_percent({"CPUPerc": "0%"}) == 0.0


def test_parse_cpu_percent_returns_none_when_missing() -> None:
    assert parse_cpu_percent({}) is None
    assert parse_cpu_percent({"CPUPerc": ""}) is None


def test_parse_cpu_percent_rejects_bool() -> None:
    assert parse_cpu_percent({"CPU": True}) is None
