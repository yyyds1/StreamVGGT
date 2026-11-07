#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_NAME="ACT"
export ABLATION_PROFILE="EE-C1_head_missing"
export SPARSE_WAYPOINT_PROGRESS_MODE="time"
export SPARSE_WAYPOINT_STEP_SCALE="${SPARSE_WAYPOINT_STEP_SCALE:-1.5}"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
