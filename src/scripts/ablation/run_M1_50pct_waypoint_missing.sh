#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_PROFILE="M1_50pct_waypoint_missing"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
