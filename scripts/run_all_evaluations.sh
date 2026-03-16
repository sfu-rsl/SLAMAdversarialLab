#!/bin/bash
# Run all SLAM evaluations
# Each evaluation runs 3 times to account for non-determinism

set -e  # Exit on error

# Change to project root directory
cd "$(dirname "$0")/.."

CONFIGS_DIR="configs/slamadverseriallab/experiments"

# KITTI experiments and algorithms
KITTI_CONFIGS=(
    "kitti/kitti_soiling_all_severities.yaml"
    "kitti/kitti_network_all_severities.yaml"
    "kitti/kitti_rain_all_severities.yaml"
    "kitti/kitti_rain_soiling_all_severities.yaml"
    "kitti/kitti_night_fog_all_severities.yaml"
    "kitti/kitti_rain_motionblur_all_severities.yaml"
    "kitti/kitti_soiling_network_all_severities.yaml"
)
KITTI_ALGORITHMS=("orbslam3" "s3pogs" "gigaslam")

# TUM experiments and algorithms
TUM_CONFIGS=(
    "tum/tum_soiling_all_severities.yaml"
    "tum/tum_network_all_severities.yaml"
    "tum/tum_crack_all_severities.yaml"
    "tum/tum_soiling_crack_all_severities.yaml"
)
TUM_ALGORITHMS=("orbslam3" "mast3rslam")

echo "=========================================="
echo "SLAM Evaluation Script"
echo "Working directory: $(pwd)"
echo "=========================================="

echo ""
echo "=========================================="
echo "KITTI Evaluations"
echo "=========================================="

for config in "${KITTI_CONFIGS[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo ""
    echo "--- $config_name ---"
    for algo in "${KITTI_ALGORITHMS[@]}"; do
        echo ">>> Running: $config with $algo"
        python -m slamadverseriallab evaluate "$CONFIGS_DIR/$config" --slam "$algo" --num-runs 3 || echo "FAILED: $config with $algo"
    done
done

echo ""
echo "=========================================="
echo "TUM Evaluations"
echo "=========================================="

for config in "${TUM_CONFIGS[@]}"; do
    config_name=$(basename "$config" .yaml)
    echo ""
    echo "--- $config_name ---"
    for algo in "${TUM_ALGORITHMS[@]}"; do
        echo ">>> Running: $config with $algo"
        python -m slamadverseriallab evaluate "$CONFIGS_DIR/$config" --slam "$algo" --num-runs 3 || echo "FAILED: $config with $algo"
    done
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
