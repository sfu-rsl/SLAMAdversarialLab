"""Unit tests for the deadline-drop-log remap helper.

These exercise the framework-level helper directly. The DROID-SLAM
trajectory-conversion code uses it under the hood; tests against the
DROID wrapper itself are in test_droidslam_deadline_remap.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slamadversariallab.runtime_stress.deadline_remap import (
    DROP_LOG_FILENAME,
    DropLog,
    load_drop_log,
    remap_internal_indices,
    remap_internal_indices_from_dir,
)


# ---------------------------------------------------------------------------
# load_drop_log
# ---------------------------------------------------------------------------

def test_load_drop_log_returns_none_when_dir_has_no_log(tmp_path):
    # Absent file is the ONLY None case: the SLAM ran without the harness, so
    # an identity remap is correct.
    assert load_drop_log(tmp_path) is None


def test_load_drop_log_raises_on_malformed_json(tmp_path):
    # A present-but-corrupt log must fail loud, not silently identity-remap.
    (tmp_path / DROP_LOG_FILENAME).write_text("not json {{")
    with pytest.raises(ValueError):
        load_drop_log(tmp_path)


def test_load_drop_log_raises_on_empty_survivors(tmp_path):
    (tmp_path / DROP_LOG_FILENAME).write_text(json.dumps({
        "survivors": [],
        "dropped": [],
        "total_items": 0,
    }))
    with pytest.raises(ValueError):
        load_drop_log(tmp_path)


def test_load_drop_log_raises_when_survivors_missing(tmp_path):
    (tmp_path / DROP_LOG_FILENAME).write_text(json.dumps({"dropped": [1, 2]}))
    with pytest.raises(ValueError):
        load_drop_log(tmp_path)


def test_load_drop_log_parses_full_payload(tmp_path):
    (tmp_path / DROP_LOG_FILENAME).write_text(json.dumps({
        "survivors": [0, 2, 5, 9],
        "dropped": [1, 3, 4, 6, 7, 8],
        "target_fps": 10.0,
        "total_items": 10,
        "warmup_frames": 2,
    }))
    log = load_drop_log(tmp_path)
    assert log is not None
    assert log.survivors == [0, 2, 5, 9]
    assert log.dropped == [1, 3, 4, 6, 7, 8]
    assert log.target_fps == 10.0
    assert log.total_items == 10
    assert log.warmup_frames == 2


def test_load_drop_log_supplies_safe_defaults_for_missing_optional_fields(tmp_path):
    (tmp_path / DROP_LOG_FILENAME).write_text(json.dumps({
        "survivors": [0, 2],
    }))
    log = load_drop_log(tmp_path)
    assert log is not None
    assert log.survivors == [0, 2]
    assert log.dropped == []
    assert log.target_fps == 0.0
    assert log.total_items == 2  # falls back to len(survivors)
    assert log.warmup_frames == 0


# ---------------------------------------------------------------------------
# remap_internal_indices
# ---------------------------------------------------------------------------

def test_remap_with_no_log_is_identity():
    assert remap_internal_indices([0, 1, 2, 3], None) == [0, 1, 2, 3]


def test_remap_empty_input_is_empty_output():
    assert remap_internal_indices([], None) == []
    assert remap_internal_indices(
        [], DropLog(survivors=[0, 1], dropped=[], target_fps=10, total_items=2)
    ) == []


def test_remap_applies_survivor_lookup():
    log = DropLog(
        survivors=[0, 3, 4, 9],
        dropped=[1, 2, 5, 6, 7, 8],
        target_fps=10.0,
        total_items=10,
    )
    # SLAM's internal counters 0..3 -> sampled-stream positions
    # via survivors lookup.
    assert remap_internal_indices([0, 1, 2, 3], log) == [0, 3, 4, 9]
    # Out-of-order internal indices remap correctly too.
    assert remap_internal_indices([3, 0, 2], log) == [9, 0, 4]


def test_remap_coerces_floats_and_numpy_like_to_int():
    log = DropLog(
        survivors=[10, 20, 30],
        dropped=[],
        target_fps=10.0,
        total_items=3,
    )
    # DROID stores tstamps as floats; helper must accept and coerce.
    assert remap_internal_indices([0.0, 1.0, 2.0], log) == [10, 20, 30]


def test_remap_raises_on_negative_internal_index():
    log = DropLog(
        survivors=[0, 1, 2],
        dropped=[],
        target_fps=10.0,
        total_items=3,
    )
    with pytest.raises(ValueError, match="non-negative"):
        remap_internal_indices([0, -1], log)


def test_remap_raises_on_out_of_range_internal_index():
    log = DropLog(
        survivors=[0, 5, 9],
        dropped=[1, 2, 3, 4, 6, 7, 8],
        target_fps=10.0,
        total_items=10,
    )
    with pytest.raises(ValueError, match="out of range"):
        remap_internal_indices([0, 1, 5], log)


# ---------------------------------------------------------------------------
# remap_internal_indices_from_dir
# ---------------------------------------------------------------------------

def test_from_dir_uses_log_when_present(tmp_path):
    (tmp_path / DROP_LOG_FILENAME).write_text(json.dumps({
        "survivors": [0, 2, 5],
        "dropped": [1, 3, 4],
    }))
    assert remap_internal_indices_from_dir([0, 1, 2], tmp_path) == [0, 2, 5]


def test_from_dir_is_identity_when_no_log(tmp_path):
    assert remap_internal_indices_from_dir([0, 1, 2], tmp_path) == [0, 1, 2]


def test_from_dir_raises_when_log_malformed(tmp_path):
    # A present-but-corrupt log fails loud rather than identity-remapping.
    (tmp_path / DROP_LOG_FILENAME).write_text("not json")
    with pytest.raises(ValueError):
        remap_internal_indices_from_dir([0, 1, 2], tmp_path)


# ---------------------------------------------------------------------------
# Re-export sanity: the helpers are accessible from the package root.
# ---------------------------------------------------------------------------

def test_helpers_exported_from_runtime_stress_package():
    from slamadversariallab import runtime_stress
    assert hasattr(runtime_stress, "DropLog")
    assert hasattr(runtime_stress, "load_drop_log")
    assert hasattr(runtime_stress, "remap_internal_indices")
    assert hasattr(runtime_stress, "remap_internal_indices_from_dir")
    assert hasattr(runtime_stress, "DROP_LOG_FILENAME")
