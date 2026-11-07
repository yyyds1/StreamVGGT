#!/usr/bin/env bash
set -euo pipefail

# Evaluate noisy waypoint training with a clean waypoint condition.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_NAME="ACT"
export ABLATION_PROFILE="EE-N0_clean_waypoint"
export SE3_GUIDANCE_NOISY_ORACLE=0
export SPARSE_WAYPOINT_PROGRESS_MODE="time"
export SPARSE_WAYPOINT_STEP_SCALE="${SPARSE_WAYPOINT_STEP_SCALE:-1.5}"

: "${CKPT_DIR:?Set CKPT_DIR to the complete noisy absolute-EE checkpoint directory}"
: "${SAVE_ROOT:?Set SAVE_ROOT to a unique output directory}"
: "${PORT:?Set PORT to the model server port}"
: "${TEST_NUM:=10}"
: "${TASK_CONFIG:=demo_clean}"

exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
