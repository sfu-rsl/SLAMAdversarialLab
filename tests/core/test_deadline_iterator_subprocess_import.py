"""Verify the SAL_RUNTIME_PATH + bare ``from deadline_iterator`` import
pattern that vendored SLAM scripts (e.g. DROID-SLAM/demo.py) rely on.

The deadline_iterator.py file sits inside the SAL package, but a
SLAM running in its own conda env doesn't have ``slamadversariallab``
installed. The framework injects the directory containing the file
via SAL_RUNTIME_PATH so the SLAM can do:

    sys.path.insert(0, os.environ["SAL_RUNTIME_PATH"])
    from deadline_iterator import DeadlineIterator

This test runs that exact pattern in a clean Python subprocess (no
inherited PYTHONPATH from pytest) to make sure it works.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DEADLINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "runtime_stress"
)


def test_deadline_iterator_importable_via_sal_runtime_path(tmp_path):
    drop_log = tmp_path / "drops.json"

    # Mirror exactly what DROID-SLAM/demo.py does: prepend SAL_RUNTIME_PATH
    # to sys.path, then bare-import deadline_iterator.
    script = f"""
import os, sys, json, time
sys.path.insert(0, os.environ['SAL_RUNTIME_PATH'])
from deadline_iterator import DeadlineIterator

it = DeadlineIterator(['F0', 'F1', 'F2', 'F3', 'F4'], target_fps=10.0)
seen = []
for item in it:
    seen.append(item)
    time.sleep(0.200)  # 200ms / frame; period is 100ms -> drops half

print(json.dumps({{
    'seen': seen,
    'survivors': it.survivors,
    'dropped': it.dropped,
}}))
"""

    env = {
        "SAL_RUNTIME_PATH": str(DEADLINE_DIR),
        "SAL_DROP_LOG_PATH": str(drop_log),
        "PATH": "/usr/bin:/bin",
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    payload = json.loads(result.stdout.strip())
    # Slow consumer at 200ms with 100ms period drops every other frame.
    assert payload["seen"] == ["F0", "F2", "F4"]
    assert payload["survivors"] == [0, 2, 4]
    assert payload["dropped"] == [1, 3]

    # Drop log written to SAL_DROP_LOG_PATH.
    assert drop_log.exists()
    with open(drop_log) as f:
        log = json.load(f)
    assert log["survivors"] == [0, 2, 4]
    assert log["dropped"] == [1, 3]
    assert log["total_items"] == 5
