"""The SAM 3 co-tenant's processor cost must be measurable.

SAM 3 is a HOST PROCESS, not a container. Container stats never see it, and the
per-workload CPU channel does not either -- that channel covers the SLAM's own
cgroup, and its `load` group reads zero procs in every co-tenant run. So without
a dedicated sampler the paper can say what the co-tenant costs on the GPU (5770
MB, 77% of the card) and NOTHING about what it costs on the processor.

That gap matters for what the co-tenant experiment is allowed to claim. The
synthetic GPU generator is fenced to 2 cores and 8 GB, so GPU-only contention
holds by construction. A real co-tenant is not fenced at all, so the same claim rests on the empirical control that the
CPU-only systems keep their pose counts beside it. A measured number turns that
inference into evidence.

These tests use real processes rather than mocks, because the thing under test is
whether /proc parsing is right.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from src.runtime_stress.load_controller import PodmanLoadController  # noqa: E402


def _controller_with(proc):
    c = PodmanLoadController.__new__(PodmanLoadController)
    c._seg_proc = proc
    c._seg_cpu_prev = None
    return c


@pytest.fixture
def busy_tree():
    """A session leader that spawns a busy child, like SAM 3's worker layout."""
    code = (
        "import subprocess,sys,time\n"
        "kid=subprocess.Popen([sys.executable,'-c','\\nwhile True: pass'])\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    time.sleep(1.0)
    yield proc
    try:
        os.killpg(os.getpgid(proc.pid), 9)
    except (ProcessLookupError, PermissionError):
        proc.kill()
    proc.wait(timeout=10)


def test_no_reading_without_two_samples(busy_tree):
    """Cores is a RATE, so the first call cannot produce one.

    Returning 0.0 here instead of None would report an idle co-tenant, which is
    the exact wrong answer: it would say the neighbour is free on the processor.
    """
    c = _controller_with(busy_tree)
    first = c._seg_cpu_sample()
    assert first["cpu_cores"] is None, "a rate needs two samples; None, never 0.0"
    assert first["procs"] >= 1


def test_it_measures_the_whole_tree_not_just_the_leader(busy_tree):
    """THE CASE THIS EXISTS FOR.

    The leader sleeps and its CHILD burns processor. Summing the leader alone
    reports ~0 cores for a co-tenant that is fully busy. The tree is found by
    session id, which start_new_session=True makes equal to the leader's pid.
    """
    c = _controller_with(busy_tree)
    c._seg_cpu_sample()
    time.sleep(2.0)
    second = c._seg_cpu_sample()

    assert second["procs"] >= 2, (
        f"expected leader plus child, found {second['procs']}: the session walk "
        "is missing descendants")
    assert second["cpu_cores"] is not None
    assert second["cpu_cores"] > 0.5, (
        f"a fully busy child must show near one core, got {second['cpu_cores']}; "
        "measuring only the sleeping leader would read ~0")
    assert second["rss_mb"] > 0


def test_it_survives_a_dead_co_tenant():
    """Teardown races must not raise. A telemetry failure must not kill a run."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    proc.wait(timeout=10)
    c = _controller_with(proc)
    out = c._seg_cpu_sample()          # must not raise
    assert out["procs"] == 0
    assert out["cpu_cores"] is None


def test_no_co_tenant_reports_nothing():
    """Scenarios without a co-tenant must add no keys at all, not zeros."""
    c = _controller_with(None)
    assert c._seg_cpu_sample() == {}
