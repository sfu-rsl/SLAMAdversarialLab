"""Per-frame handoff timing recorded by the deadline iterator.

The handoff log is the latency-side telemetry: entry is when the SLAM asked
for a frame (i.e. finished the previous one), yield when the frame was handed
over. SLAM processing time for handoff k is entry(k+1) - yield(k); pacing
sleeps occur between entry and yield and are excluded by construction. These
tests pin that contract, since an off-by-one here silently misattributes
pacing time to the SLAM.
"""
import json
import time

from slamadversariallab.runtime_stress.deadline_iterator import DeadlineIterator


def drain(it):
    out = []
    for item in it:
        out.append(item)
    return out


def test_one_handoff_per_survivor_and_monotone(monkeypatch, tmp_path):
    log = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log))
    it = DeadlineIterator(list(range(20)), target_fps=1000.0, warmup_frames=3)
    drain(it)
    payload = json.loads(log.read_text())
    assert len(payload["handoffs"]) == len(payload["survivors"])
    assert [h[0] for h in payload["handoffs"]] == payload["survivors"]
    ts = [v for h in payload["handoffs"] for v in h[1:]]
    assert ts == sorted(ts), "entry/yield timestamps must be non-decreasing"
    assert all(h[2] >= h[1] for h in payload["handoffs"]), "yield >= entry"


def test_end_entry_present_on_exhaustion(monkeypatch, tmp_path):
    log = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log))
    it = DeadlineIterator(list(range(5)), target_fps=1000.0)
    drain(it)
    payload = json.loads(log.read_text())
    assert payload["end_entry_s"] is not None
    assert payload["end_entry_s"] >= payload["handoffs"][-1][2]


def test_processing_time_excludes_pacing(monkeypatch, tmp_path):
    """A fast consumer of a slow stream must not be charged the pacing wait.

    At 20 fps a fast consumer spends ~50 ms per frame inside __next__ (the
    pacing sleep) but almost nothing between yield and its next call. The
    entry-to-yield interval carries the pacing; yield-to-next-entry (the SLAM
    time) stays tiny.
    """
    log = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log))
    it = DeadlineIterator(list(range(6)), target_fps=20.0, warmup_frames=1)
    drain(it)  # consumer cost ~0
    h = json.loads(log.read_text())["handoffs"]
    slam_times = [h[k + 1][1] - h[k][2] for k in range(len(h) - 1)]
    assert max(slam_times) < 0.02, f"pacing leaked into SLAM time: {slam_times}"
    paced_calls = [h[k][2] - h[k][1] for k in range(2, len(h))]
    assert max(paced_calls) > 0.03, "pacing should appear inside the call"


def test_drop_newest_records_handoffs(monkeypatch, tmp_path):
    log = tmp_path / "drops.json"
    monkeypatch.setenv("SAL_DROP_LOG_PATH", str(log))
    it = DeadlineIterator(list(range(30)), target_fps=200.0, warmup_frames=2,
                          queue_size=3, drop_policy="drop_newest")
    for _ in it:
        time.sleep(0.012)  # slower than the camera: forces drops
    payload = json.loads(log.read_text())
    assert payload["dropped"], "scenario should drop frames"
    assert len(payload["handoffs"]) == len(payload["survivors"])
    assert [h[0] for h in payload["handoffs"]] == payload["survivors"]
