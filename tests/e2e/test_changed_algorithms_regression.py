"""End-to-end regression wrapper for changed algorithm contracts.

This test is intentionally opt-in because it requires full external SLAM
dependencies and can take significant time.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def test_changed_algorithms_regression_script() -> None:
    """Run the changed-algorithm E2E regression script when explicitly enabled."""
    if os.environ.get("RUN_CHANGED_ALGO_E2E") != "1":
        pytest.skip("Set RUN_CHANGED_ALGO_E2E=1 to run E2E changed-algorithm regression checks.")

    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "regression" / "run_changed_algo_e2e.sh"
    assert script_path.exists(), f"Missing regression script: {script_path}"

    timeout_sec = int(os.environ.get("CHANGED_ALGO_E2E_TIMEOUT_SEC", "21600"))
    completed = subprocess.run(
        ["bash", str(script_path)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        env=os.environ.copy(),
    )

    if completed.returncode != 0:
        raise AssertionError(
            "Changed-algorithm E2E regression script failed.\n"
            f"Exit code: {completed.returncode}\n"
            f"STDOUT:\n{completed.stdout}\n"
            f"STDERR:\n{completed.stderr}\n"
        )

    assert "ALL E2E REGRESSION CHECKS PASSED" in completed.stdout
