#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${PYTHON_BIN:=$(command -v python3 || command -v python)}"
: "${NGPU:=4}"
: "${MASTER_PORT:=29527}"
: "${SAVE_ROOT:?Set a unique absolute SAVE_ROOT for this run}"
mkdir -p "$SAVE_ROOT"
(set -o noclobber; printf '%s\n' "$(date -Is)" > "$SAVE_ROOT/.launched")
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
NVIDIA_LIB_DIRS="$($PYTHON_BIN -c 'import pathlib,site;print(":".join(str(p) for b in site.getsitepackages() for p in sorted((pathlib.Path(b)/"nvidia").glob("*/lib")) if p.is_dir()))' 2>/dev/null || true)"
if [[ -n "$NVIDIA_LIB_DIRS" ]]; then export LD_LIBRARY_PATH="$NVIDIA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; fi
"$PYTHON_BIN" -c 'import torch,transformers; from transformers import Dinov2Model; print("RUNTIME",torch.__version__,transformers.__version__,"GPUs",torch.cuda.device_count(),flush=True); assert torch.cuda.device_count()==4'
"$PYTHON_BIN" -u -m torch.distributed.run --nproc_per_node="$NGPU" --master_port "$MASTER_PORT" --local-ranks-filter=0 --tee 3 -m train_va --config-name head_wrist_dinov2_clean_lora_train --save-root "$SAVE_ROOT" "$@" 2>&1 | tee "$SAVE_ROOT/train.console.log"
