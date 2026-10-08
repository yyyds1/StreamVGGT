#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
: "${NGPU:=8}"
: "${MASTER_PORT:=29512}"
: "${LOG_RANK:=0}"
: "${TORCHFT_LIGHTHOUSE:=http://localhost:29510}"
: "${CONFIG_NAME:=head_camera_temporal_train}"
: "${SAVE_ROOT:=./train_out/head_camera_stage2}"
: "${LORA:=1}"
: "${SINGLE_TASK:=}"
: "${RESUME_CHECKPOINT_DIR:=}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHFT_LIGHTHOUSE
NVIDIA_LIB_DIRS="$(python -c 'import pathlib, site; print(":".join(str(p) for base in site.getsitepackages() for p in sorted((pathlib.Path(base) / "nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [[ -n "${NVIDIA_LIB_DIRS}" ]]; then
    export LD_LIBRARY_PATH="${NVIDIA_LIB_DIRS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [[ "$LORA" == "1" || "$LORA" == "true" ]]; then
    LORA_ARGS=(--use-lora --no-freeze-visual-backbone)
elif [[ "$LORA" == "0" || "$LORA" == "false" ]]; then
    LORA_ARGS=(--no-use-lora --no-freeze-visual-backbone)
else
    echo "LORA must be 1/true (LoRA) or 0/false (full parameter training), got: $LORA" >&2
    exit 2
fi
args=(--config-name "$CONFIG_NAME" --save-root "$SAVE_ROOT" "${LORA_ARGS[@]}")
if [[ -n "$RESUME_CHECKPOINT_DIR" ]]; then
    args+=(--resume-checkpoint-dir "$RESUME_CHECKPOINT_DIR")
fi
if [[ -n "$SINGLE_TASK" ]]; then args+=(--single-task "$SINGLE_TASK"); fi
if [[ $# -gt 0 ]]; then args+=("$@"); fi
exec python -m torch.distributed.run \
    --nproc_per_node="$NGPU" --local-ranks-filter="$LOG_RANK" \
    --master_port "$MASTER_PORT" --tee 3 -m train_va "${args[@]}"
