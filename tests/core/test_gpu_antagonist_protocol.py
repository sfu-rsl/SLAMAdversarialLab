"""GPU antagonist control/status protocol, torch-free.

The antagonist's decision logic (ballast plan, control parsing, atomic
status writes) is pure stdlib and testable host-side; the torch import is
deferred inside main(). A subprocess smoke (house pattern:
test_deadline_iterator_subprocess_import.py) guards argparse/syntax.
"""

import json
import subprocess
import sys
from pathlib import Path

from slamadversariallab.runtime_stress.gpu_antagonist import (
    CHUNK_BYTES,
    ballast_chunks_for,
    parse_control,
    read_json,
    write_json_atomic,
)

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "src" / "runtime_stress" / "gpu_antagonist.py"
)


def test_ballast_chunks_for():
    assert ballast_chunks_for(0) == 0
    assert ballast_chunks_for(-5) == 0
    assert ballast_chunks_for(256) == 1
    assert ballast_chunks_for(2048) == 2048 * 1024 * 1024 // CHUNK_BYTES
    assert ballast_chunks_for(100) == 1  # small requests still hold one chunk


def test_parse_control_normalizes():
    ctl = parse_control({"epoch": 3, "vram_mb": 2048, "matmul_n": 4096, "duty_cycle": 0.5})
    assert ctl == {"epoch": 3, "vram_mb": 2048, "matmul_n": 4096, "duty_cycle": 0.5}
    # clamping + defaults
    ctl = parse_control({"epoch": 4, "duty_cycle": 7.0, "vram_mb": -3})
    assert ctl["duty_cycle"] == 1.0 and ctl["vram_mb"] == 0 and ctl["matmul_n"] == 0


def test_parse_control_rejects_unactionable():
    assert parse_control(None) == {}
    assert parse_control({"vram_mb": 100}) == {}  # no epoch
    assert parse_control({"epoch": "x", "vram_mb": "y"}) == {}


def test_status_write_read_roundtrip(tmp_path):
    p = tmp_path / "status.json"
    write_json_atomic(p, {"state": "ready", "epoch": 2})
    assert read_json(p) == {"state": "ready", "epoch": 2}
    assert not p.with_suffix(".tmp").exists()
    # partial/corrupt reads return None, never raise
    p.write_text("{not json")
    assert read_json(p) is None
    assert read_json(tmp_path / "missing.json") is None


def test_script_help_runs_without_torch():
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "--control" in result.stdout and "--status" in result.stdout
