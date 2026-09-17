"""Tests for the pipeline's pre-scenario page-cache eviction helper.

The helper lets IO-throttled scenarios actually exercise the cgroup cap:
without it, the baseline run pre-warms the dataset's page cache and
subsequent reads service from RAM rather than disk.
"""

import os
from pathlib import Path

from slamadversariallab.pipelines.runtime_stress_evaluation import (
    _evict_paths_from_page_cache,
)


def test_evict_paths_walks_files_and_returns_no_error(tmp_path: Path) -> None:
    """Must walk every regular file under each path and not raise."""
    (tmp_path / "a").write_bytes(b"x" * 8192)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"y" * 4096)

    _evict_paths_from_page_cache([tmp_path])


def test_evict_paths_skips_missing_paths_silently(tmp_path: Path) -> None:
    """Non-existent paths must be skipped without raising."""
    _evict_paths_from_page_cache([tmp_path / "does-not-exist"])


def test_evict_paths_skips_zero_byte_files(tmp_path: Path) -> None:
    """A 0-byte file must not crash the fadvise call (size==0 path)."""
    (tmp_path / "empty").write_bytes(b"")
    _evict_paths_from_page_cache([tmp_path])


def test_evict_paths_handles_unreadable_files(tmp_path: Path, monkeypatch) -> None:
    """A file we can't open (permission error) must be skipped, not raise."""
    p = tmp_path / "unreadable"
    p.write_bytes(b"x" * 1024)

    real_open = os.open

    def fake_open(path, flags, *args, **kwargs):
        if Path(path) == p:
            raise PermissionError("no read")
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", fake_open)

    _evict_paths_from_page_cache([tmp_path])
