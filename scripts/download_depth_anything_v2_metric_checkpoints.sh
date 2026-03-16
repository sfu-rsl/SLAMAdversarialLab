#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CHECKPOINT_DIR="${ROOT_DIR}/deps/depth-estimation/Depth-Anything-V2/metric_depth/checkpoints"

encoder="${DEPTH_ANYTHING_ENCODER:-vitl}"
dataset="${DEPTH_ANYTHING_DATASET:-all}"

usage() {
    cat <<'EOF'
Usage: ./scripts/download_depth_anything_v2_metric_checkpoints.sh [--encoder vits|vitb|vitl] [--dataset hypersim|vkitti|all]

Defaults:
  --encoder vitl
  --dataset all

Examples:
  ./scripts/download_depth_anything_v2_metric_checkpoints.sh
  ./scripts/download_depth_anything_v2_metric_checkpoints.sh --encoder vitb --dataset vkitti
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --encoder)
            encoder="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "${encoder}" in
    vits|vitb|vitl) ;;
    *)
        echo "Invalid encoder: ${encoder}" >&2
        usage >&2
        exit 1
        ;;
esac

case "${dataset}" in
    hypersim|vkitti|all) ;;
    *)
        echo "Invalid dataset selector: ${dataset}" >&2
        usage >&2
        exit 1
        ;;
esac

download() {
    local name="$1"
    local url="$2"
    local target="${CHECKPOINT_DIR}/${name}"

    if [[ -f "${target}" ]]; then
        echo "Already present: ${target}"
        return 0
    fi

    echo "Downloading ${name}..."
    if command -v wget >/dev/null 2>&1; then
        wget -O "${target}" "${url}"
    elif command -v curl >/dev/null 2>&1; then
        curl -L "${url}" -o "${target}"
    else
        echo "Neither wget nor curl is installed." >&2
        exit 1
    fi
}

variant_for_encoder() {
    case "$1" in
        vits) echo "Small" ;;
        vitb) echo "Base" ;;
        vitl) echo "Large" ;;
        *)
            echo "Unsupported encoder: $1" >&2
            exit 1
            ;;
    esac
}

mkdir -p "${CHECKPOINT_DIR}"

variant="$(variant_for_encoder "${encoder}")"

if [[ "${dataset}" == "hypersim" || "${dataset}" == "all" ]]; then
    download \
        "depth_anything_v2_metric_hypersim_${encoder}.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-${variant}/resolve/main/depth_anything_v2_metric_hypersim_${encoder}.pth"
fi

if [[ "${dataset}" == "vkitti" || "${dataset}" == "all" ]]; then
    download \
        "depth_anything_v2_metric_vkitti_${encoder}.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-VKITTI-${variant}/resolve/main/depth_anything_v2_metric_vkitti_${encoder}.pth"
fi

echo "Depth Anything V2 metric checkpoints are available in:"
echo "  ${CHECKPOINT_DIR}"
