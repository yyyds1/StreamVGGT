#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ABLATION_PROFILE="W1_clean_waypoint"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
