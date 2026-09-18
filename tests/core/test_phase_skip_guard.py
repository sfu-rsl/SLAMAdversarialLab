"""A phase whose controls never applied must not read as a completed phase.

Frame-anchored phase boundaries are evaluated only when a telemetry sample is
taken. If the SLAM's frame counter advances further between two samples than a
phase is wide, the orchestrator steps straight over that phase. Its controls
never actuate, and nothing in the trajectory says so -- the run completes and
looks like a successful multi-phase experiment.

Two real campaigns died this way at a 500 ms sample period:

    system      counter jump    phase window   what the trace held
    gigaslam    16 -> 116       squeeze 60-100  no `squeeze` key at all
    dpvslam     62 -> 131       squeeze 75-125  1 sample, 43% throttled
    orbslam3i   smooth          squeeze 200-350 correct; result stands

The stall guard does not cover this. There the counter never MOVES; here it
moves too far. These tests pin the distinction.
"""
from __future__ import annotations

import pytest

from src.runtime_stress.orchestrator import RuntimeStressOrchestrator


class _Phase:
    def __init__(self, name: str, until_frame: int):
        self.name = name
        self.until_frame = until_frame
        self.duration_s = 0.0


def _orchestrator(monkeypatch, frame_anchored: bool = True):
    """A bare orchestrator with the phase machinery wired and nothing else."""
    orch = RuntimeStressOrchestrator.__new__(RuntimeStressOrchestrator)
    orch.request = type("R", (), {})()
    orch.request.phases = [
        _Phase("clean_before", 60),
        _Phase("squeeze", 100),
        _Phase("released", 200),
    ]
    orch._frame_anchored = frame_anchored
    orch._current_phase_index = 0
    orch._controller_error = None
    orch._last_frame = 0
    orch.events = []
    orch._record_event = lambda *a, **k: orch.events.append((a, k))
    return orch


def test_a_stepped_over_phase_invalidates_the_run(monkeypatch):
    """THE CASE THIS EXISTS FOR -- gigaslam's, exactly.

    Counter 16 -> 116 crosses BOTH the 60 and the 100 boundary, so `squeeze`
    is entered zero times and its cap never applies. Before this guard the run
    finished, scored 11 poses, and was indistinguishable from one that had
    genuinely been squeezed and recovered.
    """
    orch = _orchestrator(monkeypatch)
    orch._check_phase_skip(2, prev_frame=16, current_frame=116, elapsed_s=22.8)

    assert orch._controller_error, "a skipped phase must invalidate the run"
    assert "squeeze" in orch._controller_error
    assert "16" in orch._controller_error and "116" in orch._controller_error, (
        "the message must carry both ends of the jump so the cause is readable")
    assert orch.events and orch.events[0][0][0] == "error"


def test_a_normal_boundary_crossing_is_silent():
    """dpvslam's corrected run reached squeeze one step at a time.

    Advancing exactly one phase is the intended behaviour and must never be
    flagged, or every healthy run would be marked invalid.
    """
    orch = _orchestrator(None)
    orch._check_phase_skip(1, prev_frame=49, current_frame=62, elapsed_s=4.9)
    assert orch._controller_error is None
    assert not orch.events


def test_wall_clock_phases_are_out_of_scope():
    """Elapsed-anchored phases advance with the sampler itself.

    They cannot skip, so the guard must not fire on them -- cap ladders are
    wall-clock and must stay untouched.
    """
    orch = _orchestrator(None, frame_anchored=False)
    orch._check_phase_skip(2, prev_frame=0, current_frame=None, elapsed_s=9.0)
    assert orch._controller_error is None


def test_the_first_error_survives():
    """A skip must not overwrite an earlier, causally prior failure."""
    orch = _orchestrator(None)
    orch._controller_error = "controller prepare failed: boom"
    orch._check_phase_skip(2, prev_frame=16, current_frame=116, elapsed_s=22.8)
    assert orch._controller_error == "controller prepare failed: boom"
    assert orch.events, "it still records the skip as an event"


def test_the_very_first_phase_application_is_not_a_skip():
    """Startup moves None -> 0 and must stay silent."""
    orch = _orchestrator(None)
    orch._current_phase_index = None
    orch._check_phase_skip(0, prev_frame=0, current_frame=0, elapsed_s=0.1)
    assert orch._controller_error is None


@pytest.mark.parametrize("target,skipped", [(2, ["squeeze"])])
def test_it_names_which_phase_was_lost(target, skipped):
    """Naming the phase is what makes the failure actionable."""
    orch = _orchestrator(None)
    orch._check_phase_skip(target, prev_frame=16, current_frame=116, elapsed_s=1.0)
    for name in skipped:
        assert name in orch._controller_error
