#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_NAME="ACT"
export ABLATION_PROFILE="EE-C2-R_right_wrist_missing"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
