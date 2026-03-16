"""Tests for shared TUM association resolution/generation helpers."""

from pathlib import Path

import pytest

from slamadverseriallab.datasets.associations import (
    generate_tum_association_with_associate_py,
    resolve_tum_association_for_orbslam3,
)


def test_orbslam3_resolver_returns_existing_association_file(tmp_path: Path) -> None:
    assoc = tmp_path / "associations.txt"
    assoc.write_text("0.0 rgb/0.png 0.0 depth/0.png\n", encoding="utf-8")

    resolved = resolve_tum_association_for_orbslam3(tmp_path, generate_if_missing=False)

    assert resolved == assoc


def test_orbslam3_resolver_uses_assoc_glob_fallback(tmp_path: Path) -> None:
    assoc = tmp_path / "custom_assoc_file.txt"
    assoc.write_text("0.0 rgb/0.png 0.0 depth/0.png\n", encoding="utf-8")

    resolved = resolve_tum_association_for_orbslam3(tmp_path, generate_if_missing=False)

    assert resolved == assoc


def test_orbslam3_associate_py_generation_writes_stdout_verbatim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rgb_file = tmp_path / "rgb.txt"
    depth_file = tmp_path / "depth.txt"
    rgb_file.write_text("0.0 rgb/0.png\n", encoding="utf-8")
    depth_file.write_text("0.0 depth/0.png\n", encoding="utf-8")

    class _Result:
        returncode = 0
        stdout = "0.0 rgb/0.png 0.0 depth/0.png\n"
        stderr = ""

    captured = {}

    def _fake_run(cmd, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["capture_output"] = capture_output
        captured["text"] = text
        captured["timeout"] = timeout
        return _Result()

    monkeypatch.setattr("slamadverseriallab.datasets.associations.subprocess.run", _fake_run)

    assoc_path = generate_tum_association_with_associate_py(
        dataset_path=tmp_path,
        rgb_file=rgb_file,
        depth_file=depth_file,
        max_diff=0.02,
    )

    assert assoc_path == tmp_path / "associations.txt"
    assert assoc_path.read_text(encoding="utf-8") == _Result.stdout
    assert "--max_difference" in captured["cmd"]
    assert "0.02" in captured["cmd"]
    assert captured["capture_output"] is True
    assert captured["text"] is True
    assert captured["timeout"] == 60

