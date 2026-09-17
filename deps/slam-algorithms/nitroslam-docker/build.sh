#!/usr/bin/env bash
# Build the Nitro-SLAM image (nitroslam:latest).
#
# Nitro-SLAM is a GPU-accelerated ORB-SLAM3 fork; this image targets the EuRoC
# stereo-inertial path (stereo_inertial_euroc), the one driver that activates
# the FastTrack / TurboMap / FastLoop GPU modules.
#
# Like the ORB-SLAM3 image, this stages the single canonical C++ deadline
# iterator (src/runtime_stress/deadline_iterator.h) into the build context so
# the Dockerfile can COPY it beside the patched stereo_inertial_euroc.cc and
# recompile that one target for the real-time deadline harness. The staged copy
# is transient and removed on exit (so use build.sh, not a raw build).
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

echo "Building nitroslam:latest with '$runtime' (CUDA 12.8 + Vulkan/Kompute; this takes a while)..."
"$runtime" build -t nitroslam:latest -f "$here/Dockerfile" "$here"
echo "Done: nitroslam:latest"
