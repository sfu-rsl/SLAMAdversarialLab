"""Tests for the runtime-stress pipeline's realtime-deadline plumbing."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from slamadversariallab.pipelines.runtime_stress_evaluation import (
    _merge_deadline_into_stress_summary,
    _realtime_env,
    _summarize_drop_log,
)
from slamadversariallab.runtime_stress.models import RealtimeDeadline


def test_realtime_env_sets_and_restores_vars(tmp_path, monkeypatch):
    monkeypatch.delenv("SAL_DEADLINE_FPS", raising=False)
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)
    monkeypatch.delenv("SAL_RUNTIME_PATH", raising=False)
    monkeypatch.delenv("SAL_DROP_LOG_PATH", raising=False)

    drop_log = tmp_path / "drops.json"
    rt = RealtimeDeadline(target_fps=10.0, warmup_frames=5)

    with _realtime_env(rt, drop_log):
        assert os.environ["SAL_DEADLINE_FPS"] == "10.0"
        assert os.environ["SAL_DEADLINE_WARMUP_FRAMES"] == "5"
        assert "runtime_stress" in os.environ["SAL_RUNTIME_PATH"]
        assert os.environ["SAL_DROP_LOG_PATH"] == str(drop_log)

    assert "SAL_DEADLINE_FPS" not in os.environ
    assert "SAL_DEADLINE_WARMUP_FRAMES" not in os.environ
    assert "SAL_RUNTIME_PATH" not in os.environ
    assert "SAL_DROP_LOG_PATH" not in os.environ


def test_realtime_env_default_warmup_zero(tmp_path, monkeypatch):
    monkeypatch.delenv("SAL_DEADLINE_WARMUP_FRAMES", raising=False)
    rt = RealtimeDeadline(target_fps=30.0)
    with _realtime_env(rt, tmp_path / "drops.json"):
        assert os.environ["SAL_DEADLINE_WARMUP_FRAMES"] == "0"


def test_realtime_env_with_none_clears_existing_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("SAL_DEADLINE_FPS", "30")
    monkeypatch.setenv("SAL_DEADLINE_WARMUP_FRAMES", "3")
    monkeypatch.setenv("SAL_RUNTIME_PATH", "/old/path")
    monkeypatch.setenv("SAL_DROP_LOG_PATH", "/old/log")

    with _realtime_env(None, tmp_path / "drops.json"):
        assert "SAL_DEADLINE_FPS" not in os.environ
        assert "SAL_DEADLINE_WARMUP_FRAMES" not in os.environ
        assert "SAL_RUNTIME_PATH" not in os.environ
        assert "SAL_DROP_LOG_PATH" not in os.environ

    # Outer values restored.
    assert os.environ["SAL_DEADLINE_FPS"] == "30"
    assert os.environ["SAL_DEADLINE_WARMUP_FRAMES"] == "3"
    assert os.environ["SAL_RUNTIME_PATH"] == "/old/path"
    assert os.environ["SAL_DROP_LOG_PATH"] == "/old/log"


def test_summarize_drop_log_returns_none_when_missing(tmp_path):
    assert _summarize_drop_log(tmp_path / "no_such_file.json") is None


def test_summarize_drop_log_returns_none_on_malformed_json(tmp_path):
    p = tmp_path / "drops.json"
    p.write_text("not valid json {{{")
    assert _summarize_drop_log(p) is None


def test_summarize_drop_log_with_no_drops(tmp_path):
    p = tmp_path / "drops.json"
    p.write_text(json.dumps({
        "survivors": [0, 1, 2, 3, 4],
        "dropped": [],
        "total_items": 5,
        "target_fps": 10.0,
    }))
    summary = _summarize_drop_log(p)
    assert summary is not None
    assert summary["drop_count"] == 0
    assert summary["drop_rate"] == 0.0
    assert summary["longest_dropped_streak"] == 0
    assert summary["longest_dropped_gap"] == 0
    assert summary["total_items"] == 5
    assert summary["target_fps"] == 10.0


def test_summarize_drop_log_computes_streak_and_gap(tmp_path):
    # Survivors at 0, 3, 4, 9 -> dropped 1, 2, 5, 6, 7, 8.
    # Longest dropped streak: 5,6,7,8 = 4 consecutive.
    # Longest dropped gap (frames missed between consecutive survivors):
    #   0 -> 3 misses [1,2]            = 2
    #   3 -> 4 misses []                = 0
    #   4 -> 9 misses [5,6,7,8]        = 4
    # Max = 4.
    p = tmp_path / "drops.json"
    p.write_text(json.dumps({
        "survivors": [0, 3, 4, 9],
        "dropped": [1, 2, 5, 6, 7, 8],
        "total_items": 10,
        "target_fps": 10.0,
    }))
    summary = _summarize_drop_log(p)
    assert summary is not None
    assert summary["drop_count"] == 6
    assert summary["drop_rate"] == pytest.approx(0.6)
    assert summary["longest_dropped_streak"] == 4
    assert summary["longest_dropped_gap"] == 4


def test_summarize_drop_log_handles_missing_total_items(tmp_path):
    """Older drop logs without total_items should still produce a summary."""
    p = tmp_path / "drops.json"
    p.write_text(json.dumps({
        "survivors": [0, 2],
        "dropped": [1, 3],
    }))
    summary = _summarize_drop_log(p)
    assert summary is not None
    # Falls back to len(survivors) + len(dropped).
    assert summary["total_items"] == 4
    assert summary["drop_count"] == 2


# ---------------------------------------------------------------------------
# _merge_deadline_into_stress_summary
# ---------------------------------------------------------------------------

def test_merge_adds_deadline_key_preserving_other_fields(tmp_path):
    """Merging keeps every existing key the orchestrator wrote and adds
    a single 'deadline' block alongside them."""
    summary_path = tmp_path / "stress_summary.json"
    existing = {
        "scenario_name": "io_100mb_per_sec_rt10",
        "target_kind": "podman_container",
        "execution_ok": True,
        "duration_s": 6.961,
        "samples": 20,
        "events": 2,
    }
    summary_path.write_text(json.dumps(existing))

    deadline = {
        "target_fps": 10.0,
        "warmup_frames": 5,
        "total_items": 40,
        "drop_count": 0,
        "drop_rate": 0.0,
        "longest_dropped_streak": 0,
        "longest_dropped_gap": 0,
    }
    _merge_deadline_into_stress_summary(tmp_path, deadline)

    with open(summary_path) as f:
        merged = json.load(f)

    # Every original key preserved.
    for k, v in existing.items():
        assert merged[k] == v
    # New deadline block present and intact.
    assert merged["deadline"] == deadline


def _force_package_log_propagation():
    """The slamadversariallab parent logger sets propagate=False in
    src/utils/logging.py; flip it back so caplog can see the records."""
    import logging
    pkg = logging.getLogger("slamadversariallab")
    prior = pkg.propagate
    pkg.propagate = True
    return pkg, prior


def test_merge_no_op_when_stress_summary_missing(tmp_path, caplog):
    """If stress_summary.json wasn't written, merging is a no-op + warns."""
    import logging
    caplog.set_level(logging.WARNING)
    pkg, prior = _force_package_log_propagation()
    try:
        _merge_deadline_into_stress_summary(tmp_path, {"drop_count": 3})
    finally:
        pkg.propagate = prior

    assert not (tmp_path / "stress_summary.json").exists()
    assert any(
        "missing" in rec.message.lower() for rec in caplog.records
    ), f"expected 'missing' warning, got: {[r.message for r in caplog.records]}"


def test_merge_no_op_when_stress_summary_malformed(tmp_path, caplog):
    """A corrupt stress_summary.json must not propagate as an exception."""
    import logging
    caplog.set_level(logging.WARNING)

    summary_path = tmp_path / "stress_summary.json"
    summary_path.write_text("not valid json {{{")

    pkg, prior = _force_package_log_propagation()
    try:
        _merge_deadline_into_stress_summary(tmp_path, {"drop_count": 3})
    finally:
        pkg.propagate = prior

    # File content unchanged.
    assert summary_path.read_text() == "not valid json {{{"
    assert any(
        "failed to read" in rec.message.lower() for rec in caplog.records
    ), f"expected 'failed to read' warning, got: {[r.message for r in caplog.records]}"


def test_merge_overwrites_existing_deadline_block(tmp_path):
    """If a 'deadline' key is already present (e.g. a re-run), the merge
    replaces it rather than nesting or appending."""
    summary_path = tmp_path / "stress_summary.json"
    summary_path.write_text(json.dumps({
        "scenario_name": "x",
        "deadline": {"drop_count": 99, "stale_field": "from_prior_run"},
    }))

    new_deadline = {"drop_count": 2, "drop_rate": 0.05}
    _merge_deadline_into_stress_summary(tmp_path, new_deadline)

    merged = json.loads(summary_path.read_text())
    assert merged["deadline"] == new_deadline
    assert "stale_field" not in merged["deadline"]


def test_no_sidecar_deadline_summary_written_anywhere(tmp_path):
    """Regression: the old deadline_drop_summary.json sidecar must not
    reappear. (We deliberately stopped writing it as part of folding
    the data into stress_summary.json.)"""
    summary_path = tmp_path / "stress_summary.json"
    summary_path.write_text(json.dumps({"scenario_name": "x"}))
    _merge_deadline_into_stress_summary(tmp_path, {"drop_count": 0})

    assert not (tmp_path / "deadline_drop_summary.json").exists()
