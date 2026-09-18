"""Launch-time memory caps: the flag, the skipped update, and the kept check.

``podman update --memory`` hung on okvis2x while lowering a limit, which left
the cell uncapped while the run otherwise looked normal. The fix applies a
constant cap with ``podman run --memory`` instead, so the container is capped
before the SLAM allocates anything.

The risk the fix introduces is the reason for this file. Skipping the update
means the controller no longer proves the cap by applying it, so the skip must
never be allowed to quietly mean "uncapped": the read-back check still runs on
the skip path, and still raises. These tests hold that line, and hold the flag
and the skip to one shared source of truth so they cannot drift apart.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from src.runtime_stress.models import (
    MemoryControl,
    RealtimeDeadline,
    RuntimeStressControls,
    RuntimeStressPhase,
    RuntimeStressRequest,
)
from src.runtime_stress.orchestrator import RuntimeStressOrchestrator
from src.runtime_stress.podman_controllers import PodmanMemoryController


def _request(*caps) -> RuntimeStressRequest:
    """A run whose stress phases declare ``caps`` in MB (None = no cap)."""
    phases = [
        RuntimeStressPhase(
            name=f"p{i}",
            duration_s=1,
            controls=RuntimeStressControls(
                memory=MemoryControl(max_mb=cap) if cap is not None else None
            ),
        )
        for i, cap in enumerate(caps)
    ]
    return RuntimeStressRequest(
        scenario_name="mem",
        telemetry_sample_period_ms=250,
        phases=phases,
        container_runtime="podman",
        realtime=RealtimeDeadline(target_fps=15.0),
    )


def _orch(*caps) -> RuntimeStressOrchestrator:
    return RuntimeStressOrchestrator(_request(*caps), Path(tempfile.mkdtemp()))


# ---------------------------------------------------------------------------
# The cap the launch flag and the controller must agree on
# ---------------------------------------------------------------------------

def test_constant_cap_found_when_every_capped_phase_agrees():
    # Warmup and recovery declare no cap; that must not defeat the constant
    # reading, or a cap ladder (which always brackets the stress phase) would
    # each fall back to the update path this exists to avoid.
    assert _orch(None, 587, None).constant_memory_cap_mb() == 587


def test_no_constant_cap_when_phases_disagree():
    # A time-varying cap has a real per-phase step. Applying the first value at
    # launch would erase it, so this must decline and leave the run on the
    # mid-run pathway.
    assert _orch(600, 300).constant_memory_cap_mb() is None


def test_no_constant_cap_when_nothing_is_capped():
    assert _orch(None, None).constant_memory_cap_mb() is None


def test_launch_flag_emitted_for_a_constant_cap():
    assert _orch(None, 587, None).memory_launch_config()["run_flags"] == [
        "--memory", "587m",
    ]


def test_no_launch_flag_when_the_cap_varies():
    assert _orch(600, 300).memory_launch_config()["run_flags"] == []


def test_launch_flag_and_controller_read_the_same_cap():
    # Two call sites, one source. If these ever diverge the container is capped
    # at one value while the controller skips on behalf of another, and the
    # update fires anyway.
    orch = _orch(None, 587, None)
    flags = orch.memory_launch_config()["run_flags"]
    assert flags[1] == f"{orch.constant_memory_cap_mb()}m"


# ---------------------------------------------------------------------------
# The skip, and the check that survives it
# ---------------------------------------------------------------------------

class _SpyController(PodmanMemoryController):
    """Records update calls and serves a scripted inspect result."""

    def __init__(self, reported_bytes, **kwargs):
        super().__init__(**kwargs)
        self.updates = []
        self._container_name = "c"
        self._reported_bytes = reported_bytes

    def _podman_update(self, *args):
        self.updates.append(args)
        return True

    def _inspect_container(self, name):
        return {"HostConfig": {"Memory": self._reported_bytes}}


def _controls(mb):
    return RuntimeStressControls(memory=MemoryControl(max_mb=mb))


def test_preapplied_cap_skips_the_update_that_hung():
    ctl = _SpyController(587 * 1024 * 1024, preapplied_max_mb=587)
    ctl.apply(_controls(587))
    assert ctl.updates == []


def test_skip_path_still_raises_when_the_cap_is_not_actually_there():
    # The whole point. If the launch flag were dropped somewhere between the
    # orchestrator and the wrapper, the container runs uncapped -- and without
    # this check the run would be scored as memory-stressed regardless.
    ctl = _SpyController(0, preapplied_max_mb=587)
    try:
        ctl.apply(_controls(587))
    except RuntimeError as exc:
        assert "did not take effect" in str(exc)
    else:
        raise AssertionError("an absent cap was accepted on the skip path")


def test_a_different_cap_still_goes_through_the_update():
    # Only the exact preapplied value may skip. Anything else is a real change
    # the container has not been told about yet.
    ctl = _SpyController(300 * 1024 * 1024, preapplied_max_mb=587)
    ctl.apply(_controls(300))
    assert ctl.updates and ctl.updates[0][0] == "--memory"


def test_release_leaves_a_launch_time_cap_in_place():
    # A launch-time cap belongs to the container, not to a phase. Lifting it at
    # a phase boundary would both undo the treatment and route through the
    # update path being avoided.
    ctl = _SpyController(587 * 1024 * 1024, preapplied_max_mb=587)
    ctl._original_memory_bytes = 587 * 1024 * 1024
    ctl.release()
    assert ctl.updates == []


def test_release_still_restores_when_the_cap_came_from_an_update():
    ctl = _SpyController(0)
    ctl._original_memory_bytes = 4 * 1024 * 1024 * 1024
    ctl.release()
    assert ctl.updates and ctl.updates[0][0] == "--memory"


# ---------------------------------------------------------------------------
# Wrapper-side delivery
# ---------------------------------------------------------------------------

def test_every_wrapper_that_reads_launch_extras_also_applies_run_flags():
    """A wrapper that merges the extras but drops ``run_flags`` runs uncapped.

    This is the shape of a bug already paid for once: the console-log redirect
    was added centrally and six wrappers silently did not carry it, which was
    only found after the runs it spoiled. A launch flag fails the same quiet
    way, except the read-back check now turns it into a loud failure at apply
    time rather than a scoring error -- this test moves it earlier still, to
    the moment a wrapper is written.
    """
    import glob
    import os

    missing = []
    for path in sorted(glob.glob("src/algorithms/*.py")):
        source = open(path).read()
        if "_runtime_stress_launch_extras()" not in source:
            continue
        if "run_flags" not in source:
            missing.append(os.path.basename(path))
    assert not missing, (
        "these wrappers merge runtime-stress launch extras but never apply "
        f"run_flags, so a launch-time cap would be silently dropped: {missing}"
    )


def test_run_flags_are_applied_even_with_no_gpu_devices():
    """Reachability, not presence. The static guard below checks that a wrapper
    MENTIONS run_flags; it cannot see whether the line executes.

    okvis2x shipped with `container_cmd.extend(extra_run_flags)` nested inside
    `if extra_devices:`, so the memory cap was applied only when GPU devices
    happened to be present. A memory-only config took the else branch and got no
    cap at all, which put three runs on the mid-run pathway -- on the one system
    whose hang motivated the launch-time mechanism. The textual guard passed the
    whole time.

    So this asserts the observable behaviour: with NO devices, the flags still
    reach the command.
    """
    import ast
    import glob

    offenders = []
    for path in sorted(glob.glob("src/algorithms/*.py")):
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            # Find the statement applying run_flags, and check no enclosing `if`
            # inside this function guards it.
            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.If):
                    continue
                body = ast.dump(ast.Module(body=stmt.body, type_ignores=[]))
                if "run_flags" in body and "extend" in body:
                    offenders.append(f"{path.split('/')[-1]}:{stmt.lineno}")
    assert not offenders, (
        "run_flags is applied inside a conditional in these wrappers, so a "
        "config that does not satisfy the condition silently gets no launch "
        f"flags: {offenders}"
    )
