#!/bin/bash
# Run metrics-only evaluation for all experiments and algorithms

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIGS_DIR="$REPO_ROOT/configs/slamadverseriallab/experiments"

# KITTI experiments and algorithms
KITTI_CONFIGS=(
    "kitti/kitti_network_all_severities.yaml"
    "kitti/kitti_night_fog_all_severities.yaml"
    "kitti/kitti_rain_all_severities.yaml"
    "kitti/kitti_rain_motionblur_all_severities.yaml"
    "kitti/kitti_rain_soiling_all_severities.yaml"
    "kitti/kitti_soiling_all_severities.yaml"
    "kitti/kitti_soiling_network_all_severities.yaml"
)
KITTI_ALGORITHMS=("s3pogs" "gigaslam" "orbslam3")

# TUM experiments and algorithms
TUM_CONFIGS=(
    "tum/tum_crack_all_severities.yaml"
    "tum/tum_network_all_severities.yaml"
    "tum/tum_soiling_all_severities.yaml"
    "tum/tum_soiling_crack_all_severities.yaml"
)
TUM_ALGORITHMS=("orbslam3" "mast3rslam" "droidslam")

echo "=== Running KITTI experiments ==="
for config in "${KITTI_CONFIGS[@]}"; do
    for algo in "${KITTI_ALGORITHMS[@]}"; do
        echo ""
        echo ">>> Running: $config with $algo"
        python -m slamadverseriallab evaluate "$CONFIGS_DIR/$config" --slam "$algo" --mode metrics-only --paper-mode || echo "FAILED: $config with $algo"
    done
done

echo ""
echo "=== Running TUM experiments ==="
for config in "${TUM_CONFIGS[@]}"; do
    for algo in "${TUM_ALGORITHMS[@]}"; do
        echo ""
        echo ">>> Running: $config with $algo"
        python -m slamadverseriallab evaluate "$CONFIGS_DIR/$config" --slam "$algo" --mode metrics-only --paper-mode || echo "FAILED: $config with $algo"
    done
done

echo ""
echo "=== All experiments completed ==="
