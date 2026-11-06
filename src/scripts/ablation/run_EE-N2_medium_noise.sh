#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export POLICY_NAME="ACT"
export ABLATION_PROFILE="EE-N2_medium_noise"
exec "${SCRIPT_DIR}/../launch_client.sh" "$@"
