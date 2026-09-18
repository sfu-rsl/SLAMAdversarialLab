"""Helpers for injecting SAL realtime-harness state into a SLAM's
Podman/Docker run command.

Each container-runtime SLAM wrapper that wants to support the SAL
deadline harness needs to propagate four pieces of state into the
container when ``SAL_DEADLINE_FPS`` is set in the host:

1. A bind-mount of the SAL ``src/runtime_stress/`` directory so the
   in-container Python entry point can
   ``from deadline_iterator import DeadlineIterator``.
2. ``SAL_DEADLINE_FPS`` (and optional ``SAL_DEADLINE_WARMUP_FRAMES`` /
   ``SAL_DEADLINE_QUEUE_SIZE``) env vars passed through to the container.
3. ``SAL_RUNTIME_PATH`` set to the container-side mount point.
4. ``SAL_DROP_LOG_PATH`` set to a file inside the SLAM's
   already-mounted output directory so the host can read the JSON
   after the run.

This module centralizes the convention so each SLAM wrapper just
calls :func:`apply_realtime_to_podman_cmd` instead of inlining ~15
lines of identical plumbing.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from .deadline_remap import DROP_LOG_FILENAME, PROGRESS_FILENAME

# Container-side path where SAL bind-mounts its runtime_stress dir.
# Arbitrary but conventional within SAL; SLAM wrappers should not
# pick their own.
SAL_RUNTIME_CONTAINER_PATH = "/sal_runtime"


def apply_realtime_to_podman_cmd(
    cmd: List[str],
    container_output_path: str,
) -> None:
    """Append SAL realtime-harness injections to a Podman/Docker
    ``run`` command list, if the harness is active.

    Reads the host environment for ``SAL_DEADLINE_FPS`` (which the
    runtime-stress pipeline sets via ``_realtime_env``). When set:

    * Bind-mounts the SAL ``src/runtime_stress/`` dir at
      :data:`SAL_RUNTIME_CONTAINER_PATH` (read-only).
    * Passes ``SAL_DEADLINE_FPS``, optional
      ``SAL_DEADLINE_WARMUP_FRAMES``, ``SAL_RUNTIME_PATH``, and
      ``SAL_DROP_LOG_PATH`` into the container as ``-e`` flags.

    When ``SAL_DEADLINE_FPS`` is unset, this is a no-op. Mutates
    ``cmd`` in place.

    Parameters
    ----------
    cmd:
        The Podman/Docker command list under construction. Must
        already contain ``run`` and not yet contain the image name
        (so additions land between options and the image).
    container_output_path:
        Container-side path of the SLAM's already-mounted output
        directory. The drop log will be written to
        ``<container_output_path>/<DROP_LOG_FILENAME>``.
    """
    host_deadline_fps = os.environ.get("SAL_DEADLINE_FPS")
    if not host_deadline_fps:
        return

    host_runtime_path = os.environ.get("SAL_RUNTIME_PATH")
    if host_runtime_path:
        cmd.extend([
            "-v",
            f"{Path(host_runtime_path).resolve()}:{SAL_RUNTIME_CONTAINER_PATH}:ro",
        ])

    cmd.extend(["-e", f"SAL_DEADLINE_FPS={host_deadline_fps}"])

    warmup = os.environ.get("SAL_DEADLINE_WARMUP_FRAMES")
    if warmup is not None:
        cmd.extend(["-e", f"SAL_DEADLINE_WARMUP_FRAMES={warmup}"])

    queue_size = os.environ.get("SAL_DEADLINE_QUEUE_SIZE")
    if queue_size is not None:
        cmd.extend(["-e", f"SAL_DEADLINE_QUEUE_SIZE={queue_size}"])

    drop_policy = os.environ.get("SAL_DEADLINE_DROP_POLICY")
    if drop_policy is not None:
        cmd.extend(["-e", f"SAL_DEADLINE_DROP_POLICY={drop_policy}"])

    cmd.extend(["-e", f"SAL_RUNTIME_PATH={SAL_RUNTIME_CONTAINER_PATH}"])
    cmd.extend([
        "-e",
        f"SAL_DROP_LOG_PATH={container_output_path}/{DROP_LOG_FILENAME}",
    ])
    cmd.extend([
        "-e",
        f"SAL_PROGRESS_PATH={container_output_path}/{PROGRESS_FILENAME}",
    ])


def apply_entrypoint_override(
    cmd: List[str],
    host_path: Path,
    container_path: str,
) -> None:
    """Bind-mount a host file over its container counterpart so edits
    to the host file take effect without rebuilding the container image.

    Used by container-runtime SLAM wrappers to override the entry-point
    script (e.g. ``demo.py`` for DROID-SLAM, ``main.py`` for VGGT-SLAM)
    that was snapshot into the image at build time. SAL hooks added to
    the host-side script reach the SLAM immediately, instead of after
    the 15-25 minute rebuild a SLAM image typically costs.

    Independent of the SAL deadline harness: applies whenever
    ``host_path`` exists, so any host-side edit to the entry script
    propagates without rebuild. Mutates ``cmd`` in place; no-op if
    ``host_path`` doesn't exist.
    """
    if not host_path.exists():
        return
    cmd.extend(["-v", f"{host_path.resolve()}:{container_path}:ro"])
