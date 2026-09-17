"""A crashed S3PO-GS must not block until the campaign timeout kills it.

S3PO-GS hangs in ``mp.Process`` cleanup after its work finishes. The wrapper
works around that by stopping the container as soon as ``Total FPS`` appears.
A CRASH never reaches that line, so the workaround cannot fire.

Under E4NW's ``warmup_frames=0`` this cost 3 of s3pogs' 6 rungs: it received 2-5
frames out of 200, evo's umeyama alignment raised ``GeometryException`` on the
degenerate trajectory, and the container then sat until the campaign timeout
killed it.

**WHERE THE RECOVERY HAS TO LIVE.** The first attempt at this fix put it after
``_stream_process_output`` returns, shortening a 7200 s wait. It never executed
even once, because the streaming loop DOES NOT RETURN: the hung container holds
the pipe open, so everything after it is unreachable. The recovery must happen
inside the ``stop_on_line`` callback. That is asserted explicitly below so the
mistake cannot come back.

The stop condition is also deliberately narrow -- a specific list of terminal
errors, not "any traceback" -- because a traceback printed from an except block
is followed by a normal finish, and stopping on those would cut short a run that
was going to succeed.

The fix must stop the container WITHOUT converting the crash into a success.
Every half is asserted here.
"""
from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import pytest

from src.algorithms.s3pogs import S3POGSAlgorithm


class _FakeProcess:
    """A container process that never exits on its own, like the real hang."""

    def __init__(self):
        self.returncode = 1
        self.killed = False
        self._alive = True

    def poll(self):
        return None if self._alive else self.returncode

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired("podman", timeout or 0)
        return self.returncode

    def kill(self):
        self.killed = True
        self._alive = False


def _algo(monkeypatch, lines):
    algo = S3POGSAlgorithm.__new__(S3POGSAlgorithm)
    algo.container_runtime = "podman"
    proc = _FakeProcess()

    monkeypatch.setattr(algo, "_spawn_streaming_process",
                        lambda *a, **k: proc, raising=False)

    def _stream(process, log_prefix, stop_on_line=None):
        for line in lines:
            if stop_on_line and stop_on_line(line):
                return True
        return False

    monkeypatch.setattr(algo, "_stream_process_output", _stream, raising=False)

    waits = []
    monkeypatch.setattr(
        algo, "_wait_for_process",
        lambda p, timeout_seconds=None: waits.append(timeout_seconds), raising=False)

    stops = []
    monkeypatch.setattr(
        algo, "_stop_container",
        lambda name, runtime="docker": stops.append(name), raising=False)
    return algo, proc, waits, stops


TRACEBACK = [
    "Number of patches passed the first-stage filtering:  20",
    "Traceback (most recent call last):",
    '  File "/usr/local/lib/python3.10/dist-packages/evo/core/geometry.py", line 35',
    "evo.core.geometry.GeometryException: Degenerate covariance rank, "
]


def test_crash_stops_container_instead_of_hanging(monkeypatch):
    algo, proc, waits, stops = _algo(monkeypatch, TRACEBACK)

    ok = algo._run_container_with_total_fps_stop(
        ["podman", "run"], "s3pogs-test", [], "S3PO-GS")

    # The crash is still a FAILURE. Stopping the hang must not launder it.
    assert ok is False, "a crashed run must not report success"
    # And the hung container must actually be stopped.
    assert stops == ["s3pogs-test"], f"container was not stopped: {stops}"
    assert proc.killed, "the hung process was never killed"
    # It must never reach the post-stream wait: the streaming loop does not
    # return while the container holds the pipe open, so recovery placed after
    # it is unreachable. A first attempt at this fix made exactly that mistake.
    assert waits == [], f"recovery must happen inside the callback, not after: {waits}"


def test_clean_completion_is_unchanged(monkeypatch):
    """The Total FPS path must keep working exactly as before."""
    algo, proc, waits, stops = _algo(
        monkeypatch, ["current keyframe 21", "Eval: Total FPS 0.194"])

    ok = algo._run_container_with_total_fps_stop(
        ["podman", "run"], "s3pogs-test", [], "S3PO-GS")

    assert ok is True
    assert stops == ["s3pogs-test"]
    # The completion path returns before ever reaching the crash-grace wait.
    assert waits == [], f"clean completion should not wait on the crash path: {waits}"


NON_FATAL_TRACEBACK = [
    "Traceback (most recent call last):",
    '  File "/opt/s3pogs/utils/retry.py", line 12, in _attempt',
    "ValueError: transient parse failure, retrying",
]


def test_non_terminal_traceback_does_not_stop_the_stream(monkeypatch):
    """A traceback that is NOT in the terminal list must not cut the run short.

    Not every traceback is fatal. One printed from an except block is followed
    by a normal finish, so stopping on any traceback would kill a run that was
    going to succeed. Only the specific terminal errors stop the stream.
    """
    algo, proc, waits, stops = _algo(
        monkeypatch, NON_FATAL_TRACEBACK + ["Eval: Total FPS 0.19"])

    ok = algo._run_container_with_total_fps_stop(
        ["podman", "run"], "s3pogs-test", [], "S3PO-GS")

    assert ok is True, "a recovered run must not be reported as a crash"
    assert stops == ["s3pogs-test"]
