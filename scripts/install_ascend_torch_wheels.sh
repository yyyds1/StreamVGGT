#!/usr/bin/env bash
# =============================================================================
# 在**已激活**的 venv 中，按顺序安装本地下载的 Ascend PyTorch whl。
#
# 用法：
#   source /path/to/StreamVGGT/.venv/bin/activate
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#   bash scripts/install_ascend_torch_wheels.sh /path/to/dir/containing_wheels
#
# 目录中应包含与当前 Python 版本、aarch64、CANN 版本匹配的 whl，例如：
#   torch-2.x.x-cp310-cp310-linux_aarch64.whl
#   torchvision-*.whl
#   torch_npu-*.whl
#
# whl 获取：昇腾社区 / 华为镜像 / CANN 安装包随附，版本须与 CANN 兼容表一致。
# =============================================================================
set -euo pipefail

WHEEL_DIR="${1:-}"
if [ -z "$WHEEL_DIR" ] || [ ! -d "$WHEEL_DIR" ]; then
  echo "用法: $0 <包含 torch/torchvision/torch_npu 的 whl 的目录>" >&2
  exit 1
fi

shopt -s nullglob
install_one() {
  local pattern="$1"
  local files=("$WHEEL_DIR"/$pattern)
  if [ ${#files[@]} -eq 0 ]; then
    echo "[install_ascend_torch_wheels] 未找到: $WHEEL_DIR/$pattern" >&2
    return 1
  fi
  if [ ${#files[@]} -gt 1 ]; then
    echo "[install_ascend_torch_wheels] 多个匹配 $pattern，请只保留一个版本:" >&2
    printf '  %s\n' "${files[@]}" >&2
    return 1
  fi
  echo "[install_ascend_torch_wheels] pip install ${files[0]}"
  pip install "${files[0]}"
}

install_one "torch-*.whl"
install_one "torchvision-*.whl"
# torchaudio 可选
ta=( "$WHEEL_DIR"/torchaudio-*.whl )
if [ ${#ta[@]} -gt 0 ] && [ -f "${ta[0]}" ]; then
  echo "[install_ascend_torch_wheels] pip install ${ta[0]}"
  pip install "${ta[0]}"
fi
install_one "torch_npu-*.whl"

python3 - <<'PY'
import torch
import torch_npu  # noqa: F401
print("torch:", torch.__version__)
print("torch_npu OK, npu count:", torch.npu.device_count())
PY
