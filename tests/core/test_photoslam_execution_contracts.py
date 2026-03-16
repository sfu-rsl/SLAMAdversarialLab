"""Execution-contract tests for Photo-SLAM runtime behavior."""

from pathlib import Path

from slamadverseriallab.algorithms.photoslam import PhotoSLAMAlgorithm


def test_photoslam_hang_after_shutdown_marks_execution_failed(monkeypatch, tmp_path: Path) -> None:
    """Photo-SLAM must fail when it hangs after printing shutdown."""
    algo = PhotoSLAMAlgorithm()
    algo.photoslam_path = tmp_path

    class _FakeStdout:
        def __init__(self) -> None:
            self._lines = ["Shutdown\n"]

        def readline(self) -> str:
            if self._lines:
                return self._lines.pop(0)
            return ""

        def __iter__(self):
            return iter(())

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()
            self.returncode = 0

        def poll(self):
            return None

    fake_process = _FakeProcess()
    monkeypatch.setattr("subprocess.Popen", lambda *args, **kwargs: fake_process)

    select_calls = {"count": 0}

    def _fake_select(read_list, _write_list, _error_list, _timeout):
        select_calls["count"] += 1
        if select_calls["count"] == 1:
            return read_list, [], []
        return [], [], []

    monkeypatch.setattr("select.select", _fake_select)

    clock = [0.0, 0.0, 200.0]

    def _fake_time() -> float:
        if clock:
            return clock.pop(0)
        return 200.0

    monkeypatch.setattr("time.time", _fake_time)

    killed = {"called": False}

    def _fake_kill(_process) -> None:
        killed["called"] = True

    monkeypatch.setattr(algo, "_kill_process_group", _fake_kill)

    result = algo._run_photoslam(["photoslam_dummy_binary"])

    assert result is False
    assert killed["called"] is True
