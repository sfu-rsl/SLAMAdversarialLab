#!/usr/bin/env bash
# Build the ORB-SLAM3 image (orbslam3:latest), including SAL real-time
# deadline support.
#
# This stages the single canonical C++ deadline iterator,
# src/runtime_stress/deadline_iterator.h, into this build context so the
# Dockerfile can COPY it next to the patched mono_kitti.cc and recompile the
# example. There is exactly one copy of the header in git (under
# src/runtime_stress, beside its Python sibling deadline_iterator.py); the copy
# placed here is transient and removed on exit. This mirrors how the Python
# SLAMs import the .py from the bind-mounted /sal_runtime: one generic
# component, plumbed into each SLAM.
#
# Usage:
#   ./build.sh            # uses docker (or $CONTAINER_RUNTIME)
#   ./build.sh podman     # uses podman
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../../.." && pwd)"
runtime="${1:-${CONTAINER_RUNTIME:-docker}}"

src_header="$repo/src/runtime_stress/deadline_iterator.h"
staged_header="$here/deadline_iterator.h"

if [[ ! -f "$src_header" ]]; then
  echo "error: canonical deadline header not found: $src_header" >&2
  exit 1
fi

cp "$src_header" "$staged_header"
trap 'rm -f "$staged_header"' EXIT

echo "Building orbslam3:latest with '$runtime' (staged deadline_iterator.h from src/runtime_stress)..."
"$runtime" build -t orbslam3:latest -f "$here/Dockerfile" "$here"
echo "Done: orbslam3:latest"
