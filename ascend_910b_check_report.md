# 昇腾 910B 环境检查报告

检查时间：2025-03-13

---

## 一、硬件与驱动状态（正常）

### 1. NPU 设备

| 项目 | 结果 |
|------|------|
| **npu-smi 版本** | 25.3.rc1 |
| **当前可见卡数** | **8 张** 910B1 |
| **单卡型号** | 910B1 |
| **单卡显存 (HBM)** | 65536 MB = **64GB**（与您记忆一致） |
| **健康状态** | 全部 **OK** |
| **温度** | 约 30–34°C，正常 |
| **功耗** | 约 90–101 W，空闲正常 |
| **当前进程** | 无（8 张卡均空闲） |

说明：您提到 16 卡，本机 `npu-smi` 当前只看到 8 张。若 16 卡是整机总规模，可能另一台机器或需检查 PCIe/拓扑；若本机应见 16 张，需排查硬件/驱动/BIOS。

### 2. 驱动与固件

- **安装路径**：`/usr/local/Ascend/`
- **驱动版本**：25.3.rc1（version.info）
- **组件**：driver、firmware、add-ons 已存在
- **npu-smi**：`/usr/local/sbin/npu-smi`，可正常执行

结论：**驱动与固件已就绪，8 张 910B1 可被系统识别且状态正常。**

---

## 二、当前缺失项（影响“直接跑训练/推理”）

### 1. CANN Toolkit 未安装

- **现象**：`/usr/local/Ascend/` 下**没有** `ascend-toolkit` 目录。
- **影响**：没有 `set_env.sh`，无法配置 `ASCEND_TOOLKIT_HOME`、`LD_LIBRARY_PATH` 等，**无法跑依赖 CANN 的应用**（PyTorch-NPU、CANN 算子、MindSpore 等）。
- **处理**：从华为昇腾社区下载与驱动 25.3 匹配的 CANN toolkit（及 910B kernel 包），安装到 `/usr/local/Ascend/`，然后：
  ```bash
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
  ```
  建议将上述 `source` 写入 `~/.bashrc` 以便登录即生效。

### 2. 未配置 CANN 相关环境变量

- **当前**：`ASCEND_HOME`、`ASCEND_TOOLKIT_HOME`、`LD_LIBRARY_PATH` 等均未设置。
- **处理**：安装 CANN toolkit 后按上面方式 `source set_env.sh` 即可。

### 3. PyTorch + torch_npu 未安装

- **当前**：`pip3 list` 中无 `torch`、`torch_npu`。
- **影响**：无法用 PyTorch 在 NPU 上跑模型。
- **处理**：在**已 source set_env.sh** 的环境下，按华为/昇腾文档安装与当前 CANN、Python 匹配的：
  - `torch`
  - `torch_npu`（whl 需与 PyTorch 版本、架构 aarch64 一致）

---

## 三、昇腾 910B 使用要点（装好 CANN 后）

### 1. 日常检查卡状态

```bash
# 查看所有 NPU 状态
npu-smi info

# 持续监控（类似 nvidia-smi watch）
watch -n 1 npu-smi info
```

### 2. 指定使用某几张卡

- **环境变量**（与 NVIDIA 的 CUDA_VISIBLE_DEVICES 类似）：
  ```bash
  export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3   # 仅使用 0–3 号卡
  ```
- 程序内若用 PyTorch-NPU，通常通过 `torch.npu.set_device(i)` 或 `device="npu:i"` 指定卡。

### 3. PyTorch 在 NPU 上跑通后的自检代码

安装好 `torch`、`torch_npu` 并 `source set_env.sh` 后，可用下面脚本验证：

```python
import torch
import torch_npu

print("PyTorch version:", torch.__version__)
print("torch_npu available:", torch_npu.npu.is_available())
print("NPU device count:", torch.npu.device_count())

# 简单计算测试
x = torch.randn(2, 2).npu()
y = torch.randn(2, 2).npu()
z = x.mm(y)
print("Result on NPU:\n", z.cpu())
```

### 4. 多卡训练（例如 8 卡）

- 使用 **torchrun / torch.distributed** 或 **accelerate** 等时，需设置：
  - `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`（或实际要用的卡号）
  - 以及各框架要求的环境变量（如 RANK、WORLD_SIZE 等）。
- CANN 与 PyTorch-NPU 版本需与官方兼容表一致，否则易报错（如 ACL 相关错误）。

---

## 四、检查清单小结

| 项目 | 状态 | 说明 |
|------|------|------|
| 910B 硬件识别 | 通过 | 8 张 910B1，健康 OK |
| 单卡 64GB 显存 | 通过 | HBM 65536 MB |
| 驱动与 npu-smi | 通过 | 25.3.rc1，工作正常 |
| CANN Toolkit | 未安装 | 需安装并 source set_env.sh |
| 环境变量 | 未配置 | 随 CANN 安装一并配置 |
| PyTorch + torch_npu | 未安装 | 需在 CANN 就绪后按版本安装 |

**结论**：  
- **硬件与驱动层面**：昇腾 910B 已就绪，单卡 64GB，当前 8 张卡可被系统识别且状态正常。  
- **应用层面**：尚不能“直接”跑训练/推理，需先安装 CANN Toolkit、配置环境变量，再安装与 CANN 匹配的 PyTorch 与 torch_npu。完成上述步骤后，可按本文第三节进行使用与验证。

---

## 五、CANN 8.5.0 安装步骤（官网）

以下步骤来自昇腾官网，适用于 **910B（Atlas A2 系列）+ aarch64**。你本机驱动已装，只需装 **Toolkit + 910B ops**。

### 1. 官网入口与文档

- **一站式下载**：[社区版资源中心](https://www.hiascend.com/zh/developer/download/community/result?module=sdk+cann) — 可按版本、产品类型、架构筛选后获取下载链接或 wget 命令。
- **快速开始**：[CANN 8.5.0 快速开始](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html)
- **离线安装说明**：[安装 CANN（离线安装）](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html)

### 2. 本机环境对应

| 项目     | 你的环境        | 说明 |
|----------|-----------------|------|
| 芯片     | 910B（Atlas A2）| 选 **Atlas A2 系列** / 910B |
| 架构     | aarch64         | 执行 `arch` 可确认 |
| 已有驱动 | 25.3.rc1        | 只需装 Toolkit + ops，不必再装驱动 |

### 3. 安装命令（aarch64，root 默认路径 /usr/local/Ascend）

**前提**：安装目录可用空间 > 10G；已具备 Python3 与 pip3（CANN 8.5.0 支持 Python 3.7–3.13）。

```bash
# 1) 下载 CANN 8.5.0 Toolkit（约 1G+）
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-toolkit_8.5.0_linux-aarch64.run"

# 2) 安装 Toolkit（默认装到 /usr/local/Ascend）
bash ./Ascend-cann-toolkit_8.5.0_linux-aarch64.run --install

# 3) 下载 910B ops 算子包（Atlas A2）
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-910b-ops_8.5.0_linux-aarch64.run"

# 4) 安装 910B ops（与 Toolkit 同路径）
bash ./Ascend-cann-910b-ops_8.5.0_linux-aarch64.run --install
```

若需指定路径，可加 `--install-path=/usr/local/Ascend`（与现有驱动同路径即可）。

### 4. 安装后验证（确认是否装成功）

在终端依次执行下面命令，全部通过即表示 CANN 装成功：

```bash
# 1) 看 CANN 目录是否出现（二选一存在即可）
ls -la /usr/local/Ascend/cann/set_env.sh 2>/dev/null || ls -la /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2) 加载环境变量（路径按上面实际选）
source /usr/local/Ascend/cann/set_env.sh 2>/dev/null || source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 3) 看环境变量是否生效
echo $ASCEND_HOME
echo $LD_LIBRARY_PATH | tr ':' '\n' | head -5

# 4) NPU 仍可被识别（与装 CANN 前一致）
npu-smi info
```

- 若能看到 `set_env.sh`、`ASCEND_HOME` 有值、`LD_LIBRARY_PATH` 含 Ascend 路径，且 `npu-smi info` 正常，说明 **Toolkit + 910B ops 已装成功**。  
- 若 `set_env.sh` 不存在，说明安装路径不对或安装未完成，需检查安装日志或重装。

**关于操作系统**：你用的安装包是 `*_linux-aarch64.run`，即 **Linux + aarch64**。只要当前系统是 **Linux**（如 Ubuntu、openEuler、CentOS、Kylin 等），就没问题；CANN 支持多种 Linux 发行版。若当前不是 Linux（例如是 Windows/macOS），则无法用该 run 包，需在 Linux 环境安装。

### 5. 配置环境变量（长期生效）

8.5.0 安装后 set_env 路径可能是 `.../cann/set_env.sh` 或 `.../ascend-toolkit/set_env.sh`，以实际存在为准：

```bash
# root 默认路径示例（二选一，按实际路径）
source /usr/local/Ascend/cann/set_env.sh
# 或
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

建议写入 `~/.bashrc`，登录即生效。

### 6. 可选：NNAL（跑 vLLM/verl 时建议装）

verl 文档要求 `source .../nnal/atb/set_env.sh`，需安装 NNAL：

```bash
wget "https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%208.5.0/Ascend-cann-nnal_8.5.0_linux-aarch64.run"
bash ./Ascend-cann-nnal_8.5.0_linux-aarch64.run --install
# 然后
source /usr/local/Ascend/nnal/atb/set_env.sh   # 可一并写入 ~/.bashrc
```

### 7. 验证（含 PyTorch 时）

```bash
npu-smi info
python3 -c "import torch; import torch_npu; print('torch_npu:', torch_npu.npu.is_available())"  # 需先装 torch、torch_npu
```

---

## 六、跑 verl 的后续配置（CANN 装好后）

在 CANN 8.5.0 + 910B ops（及可选 NNAL）装好并 `source set_env.sh` 后，按 [verl Ascend Quickstart](https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html) 依次做下面几步即可跑 verl。

### 1. 基础环境（版本严格一致）

| 软件 | 版本 |
|------|------|
| Python | >= 3.10, < 3.12 |
| CANN | == 8.5.0 |
| torch | == 2.8.0 |
| torch_npu | == 2.8.0 |

torch / torch_npu 需与 CANN、架构（aarch64）匹配，从昇腾或 PyTorch-NPU 官方渠道安装对应 whl。  
（可选）若为 x86，可加：`pip config set global.extra-index-url "https://download.pytorch.org/whl/cpu/"`。

### 2. 其他依赖

```bash
pip install torchvision==0.22.1
pip uninstall -y triton triton-ascend
pip install triton-ascend==3.2.0
pip install transformers==4.57.6   # 不要用 5.0.0 及以上，verl 不支持
```

### 3. 激活 CANN 与 NNAL（每次新终端或写进 ~/.bashrc）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 或 cann/set_env.sh，按实际路径
source /usr/local/Ascend/nnal/atb/set_env.sh
```

### 4. 安装 vLLM（源码）

```bash
git clone --depth 1 --branch v0.13.0 https://github.com/vllm-project/vllm.git
cd vllm && pip install -r requirements/build.txt
VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..
```

### 5. 安装 vLLM-Ascend（源码）

```bash
git clone -b releases/v0.13.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -r requirements.txt
export COMPILE_CUSTOM_KERNELS=1 && pip install -v -e . && cd ..
```

### 6. 安装 MindSpeed（仅在用 Megatron 后端时需要）

```bash
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed && git checkout 2.3.0_core_r0.12.1 && cd ..
git clone --depth 1 --branch core_v0.12.1 https://github.com/NVIDIA/Megatron-LM.git
pip install -e MindSpeed
pip install -e Megatron-LM
pip install mbridge
```

### 7. 安装 verl

```bash
git clone --recursive https://github.com/volcengine/verl.git
cd verl && pip install -r requirements-npu.txt && pip install -v -e . && cd ..
```

### 8. 快速验证（Qwen2.5-0.5B GRPO on GSM8K）

**1）预处理数据**

```bash
cd verl   # 进入克隆的 verl 目录
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k
```

**2）跑 1 个 epoch**

```bash
set -x
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=128 \
  data.max_prompt_length=512 \
  data.max_response_length=128 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.entropy_coeff=0.001 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=20 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=console \
  trainer.project_name=verl_grpo_example_gsm8k \
  trainer.experiment_name=qwen2_7b_function_rm \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=5 \
  trainer.total_epochs=1 "$@"
```

你当前是 8 张 910B，`trainer.n_gpus_per_node=8` 已按 8 卡写好；多机时再改 `trainer.nnodes` 并配好 RANK/MASTER_ADDR 等。

### 9. 昇腾上暂不支持的库

- **flash_attn**：不支持；用 transformers 里的 attention 即可。
- **liger-kernel**：不支持。

---

## 七、其他机器一键配置脚本

**说明**：本脚本为**非 Docker** 方式，严格按上文 **五、CANN 8.5.0 安装步骤**（wget 下载 .run 包 + `bash xxx.run --install` 离线安装）执行，与第三节安装命令、第四节安装后验证、第五节/第六节环境变量一致。

在同架构（aarch64 或 x86_64）、已装 NPU 驱动的 Linux 上，可用本仓库的一键脚本自动完成 CANN + Python 环境配置。

**脚本路径**：`ascend_910b_oneclick_setup.sh`（与本文档同目录）

**用法示例**：

```bash
# 默认工作目录 /root/test，安装 CANN + 910B ops + Python 虚拟环境（torch/torch_npu）
sudo bash ascend_910b_oneclick_setup.sh

# 指定工作目录
sudo bash ascend_910b_oneclick_setup.sh /opt/ascend_env

# 只装 CANN + 910B ops，不配 Python
CANN_ONLY=1 sudo bash ascend_910b_oneclick_setup.sh

# 同时安装 NNAL（跑 vLLM/verl 时建议）
INSTALL_NNAL=1 sudo bash ascend_910b_oneclick_setup.sh

# 本机已装 CANN，只配 Python 虚拟环境和 activate 脚本
SKIP_CANN=1 bash ascend_910b_oneclick_setup.sh /root/test
```

脚本会：检测架构 → 下载并安装 CANN Toolkit、910B ops → 可选安装 NNAL → 将 `source set_env.sh` 写入 `~/.bashrc` → 在工作目录创建 `.venv` 并安装 torch 2.8.0、torch_npu 2.8.0 → 生成 `activate_ascend_env.sh` 与 `verify_npu_simple.py` → 执行一次 NPU 快速验证。  
复制 `ascend_910b_oneclick_setup.sh`、`activate_ascend_env.sh`、`verify_npu_simple.py` 到新机器时，只需带脚本即可（脚本内会重新生成 activate 与 verify）；若只带 `ascend_910b_oneclick_setup.sh` 也可完成全量配置。

---

## 八、参考

- [昇腾社区](https://www.hiascend.com/)：驱动、CANN、torch_npu 下载与版本说明
- [CANN 8.5.0 快速开始](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/quickstart/instg_quick.html)
- [CANN 8.5.0 离线安装](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/softwareinst/instg/instg_0008.html)
- [verl Ascend Quickstart](https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html)：安装流程与快速开始
- CANN 与 PyTorch-NPU 版本需严格对应，安装前请核对兼容性表
