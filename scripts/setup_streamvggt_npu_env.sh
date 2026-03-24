#!/usr/bin/env bash
# =============================================================================
# StreamVGGT / ActionVGGT — 昇腾 910B Python 依赖安装（独立虚拟环境，默认不用 verl）
#
# 前提：
#   - 已安装 CANN Toolkit + 910B ops，且能 source set_env.sh
#   - 虚拟环境中需有与 CANN 匹配的 torch、torchvision、torch_npu（从昇腾官方 whl 安装）
#
# 一键（默认在仓库根目录创建/使用 .venv）：
#   cd /path/to/StreamVGGT
#   bash scripts/setup_streamvggt_npu_env.sh
#
# 指定 Python 解释器创建 venv（推荐与 CANN 文档一致，如 3.10 / 3.11）：
#   PYTHON=python3.10 bash scripts/setup_streamvggt_npu_env.sh
#
# 指定 venv 路径：
#   VENV=/path/to/my_venv bash scripts/setup_streamvggt_npu_env.sh
#
# 仅创建空 venv 后退出（自定义路径）：
#   bash scripts/setup_streamvggt_npu_env.sh --create-only /path/to/venv
#
# 可选：HF 镜像
#   export HF_ENDPOINT=https://hf-mirror.com
#
# 若 venv 里还没有 torch / torch_npu，任选其一：
#
# A) 本地 whl 目录（推荐，与 CANN 严格一致）：
#    export ASCEND_WHEEL_DIR=/path/to/wheels   # 内含 torch / torchvision / torch_npu 的 .whl
#    bash scripts/setup_streamvggt_npu_env.sh
#    或：source .venv/bin/activate && source .../set_env.sh && bash scripts/install_ascend_torch_wheels.sh "$ASCEND_WHEEL_DIR"
#
# B) 尝试用 pip 安装（默认版本与 verl/scripts/install_sglang_mcore_npu.sh 一致；仍须与 CANN 配套表核对）：
#    INSTALL_TORCH_NPU_PIP=1 bash scripts/setup_streamvggt_npu_env.sh
#    或覆盖版本：TORCH_VER=... TORCHVISION_VER=... TORCH_NPU_VER=... INSTALL_TORCH_NPU_PIP=1 bash ...
#
# 文档与下载：https://www.hiascend.com/ 软件 → PyTorch / CANN 版本配套说明
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ASCEND_BASE="${ASCEND_BASE:-/usr/local/Ascend}"
if [ -f "${ASCEND_BASE}/ascend-toolkit/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_BASE}/ascend-toolkit/set_env.sh"
  echo "[setup] 已 source ${ASCEND_BASE}/ascend-toolkit/set_env.sh"
elif [ -f "${ASCEND_BASE}/cann/set_env.sh" ]; then
  # shellcheck disable=SC1090
  source "${ASCEND_BASE}/cann/set_env.sh"
  echo "[setup] 已 source ${ASCEND_BASE}/cann/set_env.sh"
else
  echo "[setup] 警告: 未找到 ascend-toolkit 或 cann 的 set_env.sh，请先安装 CANN 或设置 ASCEND_BASE" >&2
fi

PYTHON="${PYTHON:-python3}"

if [ "${1:-}" = "--create-only" ] && [ -n "${2:-}" ]; then
  ONLY_VENV="$2"
  echo "[setup] 仅创建虚拟环境: $ONLY_VENV"
  "$PYTHON" -m venv "$ONLY_VENV"
  # shellcheck disable=SC1090
  source "$ONLY_VENV/bin/activate"
  pip install -U pip wheel setuptools
  echo "[setup] 完成。请安装与 CANN 匹配的 torch / torchvision / torch_npu whl，再执行:"
  echo "  VENV=$ONLY_VENV bash $REPO_ROOT/scripts/setup_streamvggt_npu_env.sh"
  exit 0
fi

# 默认：仓库内独立环境（不复用 verl）
VENV="${VENV:-$REPO_ROOT/.venv}"

if [ ! -f "$VENV/bin/activate" ]; then
  echo "[setup] 未找到虚拟环境，正在创建: $VENV"
  "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1090
source "$VENV/bin/activate"
echo "[setup] 使用虚拟环境: $VENV"

pip install -U pip wheel setuptools

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
echo "[setup] HF_ENDPOINT=$HF_ENDPOINT"

# 可选：从 PyPI + 华为 ascend 总仓库用 pip 装 torch / torchvision / torch-npu
# 说明：pip 包名是 torch-npu（连字符）；不要用 .../torch-npu/simple 子路径，该目录常无可用 wheel。
# 默认版本与 verl/scripts/install_sglang_mcore_npu.sh 一致；torch-npu 若找不到 .post2 会回退试 2.7.1（仍须与 CANN 配套表一致）
if [ "${INSTALL_TORCH_NPU_PIP:-0}" = "1" ]; then
  TORCH_VER="${TORCH_VER:-2.7.1}"
  TV_VER="${TORCHVISION_VER:-0.22.1}"
  TN_VER="${TORCH_NPU_VER:-2.7.1.post2}"
  # ascend 全量 PyPI 兼容简单（含 aarch64 whl）；勿用 .../pypi/torch-npu/simple
  ASCEND_PYPI_SIMPLE="${ASCEND_PYPI_SIMPLE:-https://mirrors.huaweicloud.com/ascend/repos/pypi/simple}"
  echo "[setup] INSTALL_TORCH_NPU_PIP=1 → torch==${TORCH_VER} torchvision==${TV_VER} torch-npu==${TN_VER}"
  pip install "torch==${TORCH_VER}" "torchvision==${TV_VER}"
  _tn_ok=0
  if pip install "torch-npu==${TN_VER}" --extra-index-url "${ASCEND_PYPI_SIMPLE}"; then
    _tn_ok=1
  elif [ "${TN_VER}" != "2.7.1" ]; then
    echo "[setup] 未找到 torch-npu==${TN_VER}，尝试 torch-npu==2.7.1（Ascend 镜像）..."
    if pip install "torch-npu==2.7.1" --extra-index-url "${ASCEND_PYPI_SIMPLE}"; then
      _tn_ok=1
    fi
  fi
  if [ "${_tn_ok}" -eq 0 ]; then
    echo "[setup] 再尝试 PyPI 官方 torch-npu==2.7.1（无 extra-index）..."
    pip install "torch-npu==2.7.1" && _tn_ok=1 || true
  fi
  if [ "${_tn_ok}" -eq 0 ]; then
    echo "[setup] pip 仍无法安装 torch-npu。请改用本地 whl: export ASCEND_WHEEL_DIR=... 或从昇腾社区下载配套 wheel。" >&2
    exit 1
  fi
fi

# 可选：目录中的 whl 按顺序安装（与 CANN 配套表一致，最稳妥）
if [ -n "${ASCEND_WHEEL_DIR:-}" ]; then
  if [ ! -d "$ASCEND_WHEEL_DIR" ]; then
    echo "[setup] 错误: ASCEND_WHEEL_DIR 不是目录: $ASCEND_WHEEL_DIR" >&2
    exit 1
  fi
  echo "[setup] 从 ASCEND_WHEEL_DIR 安装: $ASCEND_WHEEL_DIR"
  bash "$REPO_ROOT/scripts/install_ascend_torch_wheels.sh" "$ASCEND_WHEEL_DIR"
fi

# torch_npu / CANN runtime deps:
# - yaml (PyYAML): missing -> ModuleNotFoundError: No module named 'yaml'
# - decorator: some AOE/TE Python adapters import it during compile init
pip install pyyaml decorator

if python3 - <<'PY'
import sys
try:
    import torch
    import torch_npu  # noqa: F401
    import torch as T
    assert hasattr(T, "npu") and T.npu.is_available(), "NPU 不可用"
except Exception as e:
    print("[setup] torch/torch_npu 未就绪:", e)
    sys.exit(1)
PY
then
  python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
print("[setup] torch:", torch.__version__)
print("[setup] torch_npu: OK, device count =", torch.npu.device_count())
PY
else
  echo "" >&2
  echo "[setup] 错误: 当前 venv 中未检测到可用的 torch + torch_npu。" >&2
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')"
  echo "[setup] 当前 venv Python: $(python3 -V) (cp${PY_VER})" >&2
  echo "" >&2
  echo "任选一种方式安装（版本须与已装 CANN 在官网「配套表」中一致）：" >&2
  echo "" >&2
  echo "  1) 本地 whl：" >&2
  echo "     export ASCEND_WHEEL_DIR=/你存放whl的目录" >&2
  echo "     bash $REPO_ROOT/scripts/setup_streamvggt_npu_env.sh" >&2
  echo "" >&2
  echo "  2) 手动按顺序 pip install 三个 whl 后，再执行本 setup。" >&2
  echo "     source $VENV/bin/activate" >&2
  echo "     source \${ASCEND_BASE:-/usr/local/Ascend}/ascend-toolkit/set_env.sh" >&2
  echo "     bash $REPO_ROOT/scripts/install_ascend_torch_wheels.sh /path/to/wheels" >&2
  echo "" >&2
  echo "  3) 与 install_sglang_mcore_npu.sh 同版本 pip（默认 2.7.1 / 0.22.1 / 2.7.1.post2）：" >&2
  echo "     INSTALL_TORCH_NPU_PIP=1 bash $REPO_ROOT/scripts/setup_streamvggt_npu_env.sh" >&2
  echo "" >&2
  echo "  配套说明: https://www.hiascend.com/ → 开发者 → 文档 / 下载" >&2
  exit 1
fi

pip install -r "$REPO_ROOT/requirements-streamvggt-npu.txt"

echo "[setup] 完成。启动训练:"
echo "  VENV=$VENV bash $REPO_ROOT/src/scripts/run_va_posttrain_npu.sh"
