#!/usr/bin/env bash
# =============================================================================
# ActionVGGT 训练 — 昇腾 NPU（HCCL）
#
# 用法（在仓库根目录或任意路径）：
#   bash /mnt/sfs_turbo/zmz/code/StreamVGGT/src/scripts/run_va_posttrain_npu.sh
#   默认使用仓库根目录 .venv（先运行 scripts/setup_streamvggt_npu_env.sh 创建并装依赖）
#   自定义 venv：export VENV=/path/to/venv
#
# 多卡数量（默认 8）：
#   NGPU=8 bash .../run_va_posttrain_npu.sh
#
# 仅使用部分卡：
#   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
#
# 说明：
#   - 已修改 train_va.py / distributed/util.py 等以支持 npu + hccl；勿再使用 CUDA 版 run_va_posttrain.sh 直接上 NPU。
#   - PYTORCH_CUDA_ALLOC_CONF 对 NPU 无效；显存相关请查昇腾文档中的 PYTORCH_NPU_* 变量（按需要自行 export）。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/.." && pwd)"
cd "$SRC_DIR"

ASCEND_BASE="${ASCEND_BASE:-/usr/local/Ascend}"
if [ -f "${ASCEND_BASE}/ascend-toolkit/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_BASE}/ascend-toolkit/set_env.sh"
elif [ -f "${ASCEND_BASE}/cann/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_BASE}/cann/set_env.sh"
fi

VENV="${VENV:-$REPO_ROOT/.venv}"
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "$VENV/bin/activate"
else
  echo "[run_va_posttrain_npu] 错误: 未找到 VENV=$VENV，请 export VENV=含 torch_npu 的虚拟环境" >&2
  exit 1
fi

umask 007

NGPU="${NGPU:-8}"
MASTER_PORT="${MASTER_PORT:-29501}"
LOG_RANK="${LOG_RANK:-0}"
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE:-http://localhost:29510}"
CONFIG_NAME="${CONFIG_NAME:-robotwin_train}"

overrides=""
if [ $# -ne 0 ]; then
  overrides="$*"
fi

export TOKENIZERS_PARALLELISM=false

# 分布式使用 HCCL（由 train_va → init_distributed 选择）；勿设置 CUDA_VISIBLE_DEVICES
# export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

set -x
TORCHFT_LIGHTHOUSE="${TORCHFT_LIGHTHOUSE}" \
python -m torch.distributed.run \
  --nproc_per_node="${NGPU}" \
  --local-ranks-filter="${LOG_RANK}" \
  --master_port "${MASTER_PORT}" \
  --tee 3 \
  -m train_va --config-name "${CONFIG_NAME}" ${overrides}
