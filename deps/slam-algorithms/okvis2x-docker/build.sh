#!/usr/bin/env bash
# Build the OKVIS2-X image (okvis2x:latest).
#
# Stages the canonical SAL C++ deadline iterator
# (src/runtime_stress/deadline_iterator.h, single source of truth beside its
# Python sibling) into this build context so the deadline layers can COPY it;
# the staged copy is transient and removed on exit. Build with this script,
# not a raw `docker build`.
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

echo "Building okvis2x:latest with '$runtime'..."
"$runtime" build -t okvis2x:latest -f "$here/Dockerfile" "$here"
echo "Done: okvis2x:latest"
