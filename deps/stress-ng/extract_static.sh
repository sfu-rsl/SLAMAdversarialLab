#!/bin/bash
# Provenance: extracts the statically linked stress-ng binary from the
# official OCI image (Alpine-based, musl-static, runs on any x86_64 Linux
# image regardless of libc). Used by the in-container load mode, which
# bind-mounts it at /sal/stress-ng inside SLAM containers at launch.
set -euo pipefail
cd "$(dirname "$0")"
IMG=ghcr.io/colinianking/stress-ng:latest
podman pull "$IMG"
cid=$(podman create "$IMG")
podman cp "$cid:/usr/bin/stress-ng" stress-ng-static
podman rm "$cid" >/dev/null
chmod +x stress-ng-static
./stress-ng-static --version
