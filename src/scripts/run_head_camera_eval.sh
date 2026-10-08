#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${PORT:=1106}"
: "${CHECKPOINT_DIR:?Set CHECKPOINT_DIR to a checkpoint_step_* directory}"
: "${PYTHON_BIN:=python3}"
# va_server loads CHECKPOINT_DIR/training_config.json unless EVAL_CONFIG is explicit.
args=(--checkpoint-dir "$CHECKPOINT_DIR" --port "$PORT")
if [[ -n "${EVAL_CONFIG:-}" ]]; then args+=(--eval-config "$EVAL_CONFIG"); fi
if [[ $# -gt 0 ]]; then args+=("$@"); fi
exec "$PYTHON_BIN" va_server.py "${args[@]}"
