#!/usr/bin/env bash
# Build HAMi-core's libvgpu.so and install it where SLAMSqueezeBench looks.
#
#   scripts/build_hami.sh                      # container build, default destination
#   scripts/build_hami.sh --dest /opt/hami/libvgpu.so
#   scripts/build_hami.sh --host               # build on the host instead
#
# The default ref is the commit the GPU controller's byte offsets are derived
# from. Building a different one makes GpuHamiController._verify_version warn.
set -euo pipefail

HAMI_CORE_REF="${HAMI_CORE_REF:-94fff568c1ccb32cdbd0f2b51de6d12d2902f074}"
CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.9.1-cudnn-devel-ubuntu20.04}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$REPO_ROOT/checkpoints/hami/libvgpu.so}"
RUNTIME=""
MODE="container"

usage() { sed -n '2,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [ $# -gt 0 ]; do
    case "$1" in
        --ref)        HAMI_CORE_REF="$2"; shift 2 ;;
        --dest)       DEST="$2"; shift 2 ;;
        --runtime)    RUNTIME="$2"; shift 2 ;;
        --cuda-image) CUDA_IMAGE="$2"; shift 2 ;;
        --host)       MODE="host"; shift ;;
        -h|--help)    usage 0 ;;
        *)            echo "unknown argument: $1" >&2; usage 1 ;;
    esac
done

# Write through sudo only when the destination actually needs it.
install_lib() {
    local src="$1" dst="$2" dir
    dir="$(dirname "$dst")"
    if mkdir -p "$dir" 2>/dev/null && [ -w "$dir" ]; then
        cp "$src" "$dst"
        chmod 644 "$dst"
    else
        echo "==> $dir needs root, using sudo"
        sudo mkdir -p "$dir"
        sudo cp "$src" "$dst"
        sudo chmod 644 "$dst"
    fi
}

if [ "$MODE" = "host" ]; then
    command -v make >/dev/null || { echo "make is required for --host" >&2; exit 1; }
    src_dir="$(mktemp -d)"
    trap 'rm -rf "$src_dir"' EXIT
    echo "==> cloning HAMi-core at $HAMI_CORE_REF"
    git clone --quiet https://github.com/Project-HAMi/HAMi-core.git "$src_dir"
    git -C "$src_dir" checkout --quiet "$HAMI_CORE_REF"
    git -C "$src_dir" apply "$REPO_ROOT/docker/hami/libvgpu-dlsym-self-handle.patch"
    echo "==> building (needs the CUDA devel headers on this host)"
    mkdir -p "$src_dir/build"
    (cd "$src_dir/build" && cmake .. \
        -DDLSYM_HOOK_ENABLE=1 -DMULTIPROCESS_LIMIT_ENABLE=1 \
        -DHOOK_MEMINFO_ENABLE=1 -DHOOK_NVML_ENABLE=1 \
        -DCMAKE_BUILD_TYPE=Debug >/dev/null && make -j"$(nproc)" vgpu)
    install_lib "$src_dir/build/libvgpu.so" "$DEST"
else
    if [ -z "$RUNTIME" ]; then
        for c in podman docker; do
            command -v "$c" >/dev/null && { RUNTIME="$c"; break; }
        done
    fi
    [ -n "$RUNTIME" ] || { echo "podman or docker is required" >&2; exit 1; }

    image="sal-hami-builder"
    container="sal-hami-extract-$$"
    echo "==> building $image with $RUNTIME at HAMi-core $HAMI_CORE_REF"
    "$RUNTIME" build \
        --build-arg "HAMI_CORE_REF=$HAMI_CORE_REF" \
        --build-arg "CUDA_IMAGE=$CUDA_IMAGE" \
        -t "$image" -f "$REPO_ROOT/docker/hami/Dockerfile" "$REPO_ROOT"

    trap '"$RUNTIME" rm -f "$container" >/dev/null 2>&1 || true' EXIT
    staged="$(mktemp -d)/libvgpu.so"
    "$RUNTIME" create --name "$container" "$image" >/dev/null
    "$RUNTIME" cp "$container:/out/libvgpu.so" "$staged"
    install_lib "$staged" "$DEST"
fi

echo
echo "installed: $DEST"
echo "point the framework at it with:"
echo "  export SAL_HAMI_LIB_HOST_PATH=\"$DEST\""
