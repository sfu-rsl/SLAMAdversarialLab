#!/bin/bash
# Run KITTI soiling experiment evaluation for all 3 SLAM algorithms

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG="configs/slamadverseriallab/experiments/kitti/kitti_soiling_all_severities.yaml"
ALGORITHMS=("orbslam3" "gigaslam" "s3pogs")

cd "$REPO_ROOT"

echo "=========================================="
echo "KITTI Soiling Experiment Evaluation"
echo "=========================================="

for algo in "${ALGORITHMS[@]}"; do
    echo ""
    echo "Running evaluation for: $algo"
    echo "------------------------------------------"
    python -m slamadverseriallab evaluate "$CONFIG" --slam "$algo" --mode metrics-only --paper
done

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="
