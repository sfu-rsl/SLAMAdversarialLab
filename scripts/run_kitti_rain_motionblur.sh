#!/bin/bash
# Run KITTI Rain + Motion Blur evaluation for all algorithms

set -e

cd "$(dirname "$0")/.."

CONFIG="configs/slamadverseriallab/experiments/kitti/kitti_rain_motionblur_all_severities.yaml"
ALGORITHMS=("orbslam3" "s3pogs" "gigaslam")

echo "=========================================="
echo "KITTI Rain + Motion Blur Evaluation"
echo "=========================================="

for algo in "${ALGORITHMS[@]}"; do
    echo ""
    echo ">>> Running: $algo"
    python -m slamadverseriallab evaluate "$CONFIG" --slam "$algo" --num-runs 3 || echo "FAILED: $algo"
done

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
