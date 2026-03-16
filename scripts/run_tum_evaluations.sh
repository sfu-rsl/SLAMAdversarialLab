#!/bin/bash
# Run TUM perturbation pipeline and SLAM evaluations
# Each evaluation runs 3 times to account for non-determinism

set -e  # Exit on error

# Change to project root directory
cd "$(dirname "$0")/.."

CONFIGS_DIR="configs/slamadverseriallab/experiments"

# TUM experiments and algorithms
TUM_CONFIGS=(
    "tum/tum_crack_all_severities.yaml"
    "tum/tum_soiling_crack_all_severities.yaml"
)
TUM_ALGORITHMS=("orbslam3" "mast3rslam")

echo ""
echo "=========================================="
echo "Step 2: Running SLAM Evaluations"
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
echo "TUM pipeline complete!"
echo "=========================================="
