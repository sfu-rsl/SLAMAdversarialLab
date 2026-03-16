#!/usr/bin/env bash
# End-to-end no-regression checks for recently changed SLAM algorithms.
# Targets: gigaslam, s3pogs, vggtslam (orbslam3/mast3rslam excluded by design).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="results/regression/logs"
mkdir -p "$LOG_DIR"

readonly PERTURBATION_NAME="night_fog_blur"
readonly STEP_TIMEOUT_SEC="${STEP_TIMEOUT_SEC:-900}"
readonly TIMEOUT_KILL_AFTER_SEC="${TIMEOUT_KILL_AFTER_SEC:-30}"

assert_file_exists() {
    local file_path="$1"
    if [[ ! -f "$file_path" ]]; then
        echo "ASSERTION FAILED: expected file not found: $file_path" >&2
        exit 1
    fi
}

assert_dir_exists() {
    local dir_path="$1"
    if [[ ! -d "$dir_path" ]]; then
        echo "ASSERTION FAILED: expected directory not found: $dir_path" >&2
        exit 1
    fi
}

assert_line_count_in_range() {
    local file_path="$1"
    local min_lines="$2"
    local max_lines="$3"
    local line_count
    line_count="$(grep -cv '^[[:space:]]*$' "$file_path" || true)"
    if (( line_count < min_lines || line_count > max_lines )); then
        echo "ASSERTION FAILED: $file_path has $line_count non-empty lines; expected [$min_lines, $max_lines]" >&2
        exit 1
    fi
}

assert_log_not_contains() {
    local log_path="$1"
    local needle="$2"
    if grep -Fq "$needle" "$log_path"; then
        echo "ASSERTION FAILED: log contains forbidden text '$needle': $log_path" >&2
        exit 1
    fi
}

run_with_timeout_and_log() {
    local log_path="$1"
    shift

    if command -v timeout >/dev/null 2>&1; then
        timeout --signal=TERM --kill-after="$TIMEOUT_KILL_AFTER_SEC" "$STEP_TIMEOUT_SEC" "$@" 2>&1 | tee "$log_path"
    else
        "$@" 2>&1 | tee "$log_path"
    fi
}

run_perturb_pipeline_once() {
    local config_path="$1"
    local experiment_name="$2"
    local run_log="$LOG_DIR/${experiment_name}_run.log"

    if [[ -n "${_RAN_EXPERIMENTS[$experiment_name]:-}" ]]; then
        return
    fi

    echo ""
    echo "=== [RUN] $experiment_name ==="
    run_with_timeout_and_log "$run_log" python -m slamadverseriallab run "$config_path"

    assert_dir_exists "results/regression/$experiment_name/images/$PERTURBATION_NAME/image_2"
    _RAN_EXPERIMENTS["$experiment_name"]=1
}

run_eval_and_assert() {
    local config_path="$1"
    local experiment_name="$2"
    local algo="$3"
    local eval_log="$LOG_DIR/${experiment_name}_${algo}_eval.log"

    echo ""
    echo "=== [EVAL] $experiment_name :: $algo ==="
    run_with_timeout_and_log \
        "$eval_log" \
        python -m slamadverseriallab evaluate "$config_path" --slam "$algo" --mode slam-only

    local traj_dir="results/regression/$experiment_name/slam_results/$algo/trajectories/run_0"
    local baseline_traj="$traj_dir/baseline.txt"
    local perturbed_traj="$traj_dir/${PERTURBATION_NAME}.txt"

    assert_dir_exists "$traj_dir"
    assert_file_exists "$baseline_traj"
    assert_file_exists "$perturbed_traj"
    assert_line_count_in_range "$baseline_traj" 1 5
    assert_line_count_in_range "$perturbed_traj" 1 5

    # Fallbacks we explicitly do not want in changed algorithms.
    assert_log_not_contains "$eval_log" "using frame indices as timestamps"
    assert_log_not_contains "$eval_log" "Could not find times.txt, using frame IDs as timestamps"
    assert_log_not_contains "$eval_log" "data.csv not found, using frame IDs as timestamps"
    assert_log_not_contains "$eval_log" "Missing left camera path in request extras"
}

declare -A _RAN_EXPERIMENTS=()

CASES=(
    "configs/slamadverseriallab/regression/kitti_5f_night_fog_blur.yaml|regression_kitti_5f_night_fog_blur|gigaslam"
    "configs/slamadverseriallab/regression/kitti_5f_night_fog_blur.yaml|regression_kitti_5f_night_fog_blur|s3pogs"
    "configs/slamadverseriallab/regression/euroc_5f_night_fog_blur.yaml|regression_euroc_5f_night_fog_blur|vggtslam"
)

if [[ "${INCLUDE_TUM_SMOKE:-0}" == "1" ]]; then
    CASES+=("configs/slamadverseriallab/regression/tum_5f_night_fog_blur.yaml|regression_tum_5f_night_fog_blur|droidslam")
fi

for case_def in "${CASES[@]}"; do
    IFS='|' read -r config_path experiment_name algo <<< "$case_def"
    run_perturb_pipeline_once "$config_path" "$experiment_name"
    run_eval_and_assert "$config_path" "$experiment_name" "$algo"
done

echo ""
echo "ALL E2E REGRESSION CHECKS PASSED"
