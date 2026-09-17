"""Page-cache eviction shared by the runtime-stress pipeline and tools/evict_cache.py.

posix_fadvise(POSIX_FADV_DONTNEED) evicts a file's cached pages from the
kernel page cache without sudo (for files we can open). Subsequent reads
must go through the block device, which is what makes cgroup IO caps
actually bite instead of serving the dataset from RAM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple


def evict(path: Path) -> Tuple[int, int]:
    """Walk ``path``; fadvise DONTNEED on every regular file.

    Returns (n_files, total_bytes) evicted. Files that cannot be opened
    are skipped silently (permission, vanished mid-walk).
    """
    n_files = 0
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            p = Path(root) / name
            try:
                fd = os.open(p, os.O_RDONLY)
            except OSError:
                continue
            try:
                size = os.fstat(fd).st_size
                if size > 0:
                    os.posix_fadvise(fd, 0, size, os.POSIX_FADV_DONTNEED)
                    total_bytes += size
                    n_files += 1
            finally:
                os.close(fd)
    return n_files, total_bytes
