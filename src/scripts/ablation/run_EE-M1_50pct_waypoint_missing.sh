#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_NAME="ACT"
export ABLATION_PROFILE="EE-M1_50pct_waypoint_missing"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
