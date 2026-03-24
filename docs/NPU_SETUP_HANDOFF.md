# StreamVGGT / ActionVGGT — 昇腾 910B 环境与交接说明

本文档汇总截至目前的 **NPU 适配、安装脚本、验证方式与代码改动**，便于后续同事接手。

---

## 1. 背景与目标

- 仓库原为 **CUDA + PyTorch** 训练流程（`README.md` / `requirements.txt`）。
- 目标：在 **华为昇腾 910B（aarch64）** 上安装依赖、跑通 **ActionVGGT** 训练入口 `train_va`，并与现有 **CANN + torch_npu** 栈对齐。
- **独立虚拟环境**：默认使用仓库内 **`.venv`**，不依赖 `verl/.venv310`（仍可手动 `export VENV=...` 覆盖）。

---

## 2. 新增与修改的文件（按用途）

### 2.1 环境与依赖

| 路径 | 说明 |
|------|------|
| `scripts/setup_streamvggt_npu_env.sh` | 创建/使用 `.venv`，`source` CANN `set_env.sh`，升级 pip；可选 `INSTALL_TORCH_NPU_PIP=1` 安装 torch/torchvision/`torch-npu`；可选 `ASCEND_WHEEL_DIR` 从本地 whl 安装；安装前自动 `pip install pyyaml`（`torch_npu` 运行时需要 `import yaml`）；最后 `pip install -r requirements-streamvggt-npu.txt` |
| `requirements-streamvggt-npu.txt` | 与根目录 `requirements.txt` 类似，但 **不含** `torch` / `torchvision`（由 CANN 配套 whl 或 pip 安装）；含 **`pyyaml`** |
| `scripts/install_ascend_torch_wheels.sh` | 按顺序安装目录内 `torch-*.whl`、`torchvision-*.whl`、可选 `torchaudio-*.whl`、`torch_npu-*.whl` |
| `src/scripts/run_va_posttrain_npu.sh` | NPU 训练启动：先 `source` Ascend 与 `.venv`，再 `torch.distributed.run` + `train_va`（默认 `VENV=$REPO_ROOT/.venv`） |
| `.gitignore` | 增加 `.venv/` |

### 2.2 验证脚本（不依赖预训练权重）

| 路径 | 说明 |
|------|------|
| `scripts/verify_checkpoint_load.py` | 按 `train_va` 方式构建 **ActionVGGT + RDT**，尝试 **load** 配置中的 `transformer_pretrained` / `action_head_pretrained`（路径相对 **`src/`** 解析为 `../ckpt/...` 即仓库 `ckpt/`）。缺文件会 SKIP 对应段 |
| `scripts/verify_forward_npu.py` | **随机初始化、不 load 权重**，构造与训练一致的 dummy batch，跑 **ActionVGGT** 前向；默认只跑 backbone，可选 `--with-rdt` 再跑 RDT（测试时补了 `state_c=0`，因 `RDT.forward` 要求 `state_c`，与当前 `train_va` 是否完全一致需另查） |

### 2.3 训练相关代码改动（NPU / HCCL）

| 文件 | 改动摘要 |
|------|----------|
| `src/distributed/util.py` | NPU 可用时：`torch.npu.set_device` + `dist.init_process_group(backend="hccl")`；否则保持 CUDA + NCCL；新增 `device_synchronize`、`device_empty_cache`；可选 `import torch_npu` |
| `src/train_va.py` | `import torch_npu`；设备 `npu:{local_rank}`；`AdamW(..., fused=仅 CUDA)`；同步/清缓存用 `device_*`；不再写死 `torch.cuda.*` |
| `src/distributed/fsdp.py` | `free_model` 使用 `device_empty_cache()` |
| `src/actionvggt/models/actionvggt.py`、`src/vggt/models/vggt.py`、`src/streamvggt/models/streamvggt.py` | `torch.cuda.amp.autocast(enabled=False)` 改为 `nullcontext()`，避免绑 CUDA |
| `src/utils/utils.py` | `save_async` 中对 `npu` 张量同样 `.cpu()` |

**未改动的路径**：`train.py`、`finetune.py`、`eval/**` 等仍以 CUDA 为主；若只在 NPU 上跑 **`train_va`**，以上足够。

---

## 3. 环境与版本要点

1. **CANN**：机器需已装 **Ascend Toolkit**（或 `cann`）并能 `source .../set_env.sh`。
2. **PyTorch NPU**：需与 CANN 配套；实践中与 `verl/scripts/install_sglang_mcore_npu.sh` 对齐的一组为 **torch 2.7.1、torchvision 0.22.1、torch-npu 2.7.1.post2**（以当前 CANN 官方配套表为准）。
3. **`torch-npu` pip 注意**：
   - 包名为 **`torch-npu`**（连字符），`import` 仍为 `torch_npu`。
   - 不要用仅 **`.../pypi/torch-npu/simple`** 子路径（易出现 `versions: none`）；应使用 **`--extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi/simple`**，或与阿里云等主索引组合。
4. **PyYAML**：安装 `torch_npu` 后务必 **`pip install pyyaml`**，否则 `import torch_npu` 可能报 `No module named 'yaml'`。
5. **`easydict`**：`src/configs/*.py` 使用 **`EasyDict`**；`requirements.txt` / `requirements-streamvggt-npu.txt` 已包含，若手工装依赖需 **`pip install easydict`**，否则 `verify_forward_npu.py` 会报 `No module named 'easydict'`。
6. **`decorator`**：部分 CANN/AOE Python 适配器初始化会依赖该包；缺失时常见报错为 `Failed to import Python module ... No module named 'decorator'`，需补装 **`pip install decorator`**。
7. **Python**：wheel 多为 **cp310** 等固定小版本，需与下载的 whl 一致（例如 Python 3.10）。

---

## 4. 推荐操作顺序（接手人）

```bash
cd /path/to/StreamVGGT
# 已手动装好 torch / torch_npu 时可省略 INSTALL_TORCH_NPU_PIP
INSTALL_TORCH_NPU_PIP=1 bash scripts/setup_streamvggt_npu_env.sh

source .venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh

python scripts/verify_forward_npu.py
python scripts/verify_checkpoint_load.py   # 放好 ckpt 后测 load

bash src/scripts/run_va_posttrain_npu.sh   # 需数据与权重、配置路径正确
```

---

## 5. 权重与数据（训练前）

- 默认配置 **`robotwin_train`**（见 `src/configs/__init__.py`）使用 **`va_robotwin_train_rdt_cfg`**：
  - 视觉侧：`transformer_pretrained` → 默认 **`../ckpt/actionvggt.pth`**（相对 `src/`，即仓库 **`ckpt/actionvggt.pth`**）
  - 动作头：`action_head_pretrained` → **`../ckpt/RDT.pth`**
- 数据集默认路径等在 `va_robotwin_train_cfg.py`（如 `dataset_path`），无 NAS 时需改成本地 LeRobot 路径及 **`empty_emb.pt`**。
- **两个权重并非强制**：不配则对应模块随机初始化，但不符合官方 post-train 设定。

---

## 6. 已知问题与后续可跟进项

1. **`train_va` 中 `action_head(...)` 调用**：`src/rdt/model.py` 的 `forward` 要求 **`state_c` 非空**，而 `train_va` 里当前调用未传 `state_c`；若训练在 RDT 处报错，需对照官方 RDT2 / 本仓库历史提交核对接口。
2. **`verify_forward_npu.py --with-rdt`**：为冒烟测试手动传入 **`state_c=zeros`**，仅用于设备/算子验证，不代表训练逻辑已对齐。
3. **依赖冲突**：`gsplat` 等可能对 aarch64/NPU 不友好；若 `pip install -r requirements-streamvggt-npu.txt` 失败，需逐项放宽或跳过非训练必需包。
4. **多卡**：脚本使用 **HCCL**；单机多卡注意 **`NGPU`**、`ASCEND_RT_VISIBLE_DEVICES` 与 `torchrun` 参数。

---

## 7. 参考文档（机器侧）

- 仓库内：`ascend_910b_check_report.md`（驱动/CANN/verl 相关说明）
- 华为昇腾：CANN 与 PyTorch 版本配套表、torch_npu 安装说明

---

*文档随仓库维护；若流程或默认版本变更，请同步更新本节。*
