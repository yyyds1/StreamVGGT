# ActionVGGT + RoboTwin 协作实验手册

本文档约定 ActionVGGT（StreamVGGT）与 `robotwin-labeled` 的目录关系、训练/评测入口、数据来源和消融实验命名。所有命令均使用仓库相对路径或环境变量，不依赖特定机器目录。

## 1. 目录与环境变量

建议将两个仓库放在同一父目录：

```text
workdir/
├── StreamVGGT/
└── robotwin-labeled/
```

在 `StreamVGGT` 根目录执行：

```bash
export STREAMVGGT_ROOT="$PWD"
export ROBOTWIN_REPO="$(cd ../robotwin-labeled && pwd)"
export DATA_ROOT="$ROBOTWIN_REPO/data_clean_test50"
export CHECKPOINT_DIR="$STREAMVGGT_ROOT/train_out/checkpoint_step_28000"
```

如果目录不在同一父目录，直接把变量改为对应 checkout 的绝对路径即可；变量只存在于运行环境，代码和文档不写死机器路径。

### ActionVGGT 环境

```bash
conda create -n ActionVGGT python=3.11 cmake=3.14.0
conda activate ActionVGGT
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

RoboTwin 环境按 `robotwin-labeled/README.md` 和官方安装文档配置。仿真资产安装完成后，建议在 RoboTwin 环境中运行评测客户端，在 ActionVGGT 环境中运行模型服务；若使用同一环境，确保 `torch`、SAPIEN/Vulkan 和项目依赖版本兼容。

## 2. 数据与模型仓库

- RoboTwin 2.0 代码与文档：[官方仓库](https://github.com/RoboTwin-Platform/RoboTwin)、[安装文档](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)。
- RoboTwin 2.0 数据集：[Hugging Face 数据集](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0/tree/main/dataset)。
- ActionVGGT 训练数据（LeRobot 格式）：[robbyant/robotwin-clean-and-aug-lerobot](https://huggingface.co/datasets/robbyant/robotwin-clean-and-aug-lerobot)。

```bash
huggingface-cli download --repo-type dataset \
  robbyant/robotwin-clean-and-aug-lerobot \
  --local-dir "$STREAMVGGT_ROOT/dataset/robotwin-clean-and-aug-lerobot"
```

- StreamVGGT 预训练权重：[Hugging Face](https://huggingface.co/lch01/StreamVGGT/)、[清华云](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/)。
- RDT2 权重：[Hugging Face](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)。

下载后通过配置文件或命令行参数指向 `$STREAMVGGT_ROOT/ckpt`、`$STREAMVGGT_ROOT/dataset` 等仓库内目录。不要把 NAS、本机 home 或容器挂载点写入提交的脚本和文档。

## 3. 训练入口

训练脚本会从当前仓库的配置和数据路径读取参数。典型多卡训练：

```bash
cd "$STREAMVGGT_ROOT"
NGPU=8 SAVE_ROOT=./train_out SINGLE_TASK= \
  bash src/scripts/run_va_train.sh
```

指定任务、配置或额外参数：

```bash
CONFIG_NAME=vga_robotwin_train SINGLE_TASK=adjust_bottle \
  bash src/scripts/run_va_train.sh
```

时间建模分支的两阶段入口：

```bash
NGPU=8 SAVE_ROOT=./train_out/head_camera_stage1 \
  bash src/scripts/run_head_camera_stage1.sh

RESUME_CHECKPOINT_DIR=./train_out/head_camera_stage1/checkpoint_step_2000 \
  LORA=1 bash src/scripts/run_head_camera_stage2.sh
```

提交训练前先运行 README 中列出的静态检查和 `compileall` 检查。训练产物应包含完整的 `checkpoint_step_*`、`training_config.json`、`transformer/` 和 `action_head/`。

## 4. 测试入口与评测约定

模型服务端：

```bash
cd "$STREAMVGGT_ROOT"
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  bash src/scripts/launch_server.sh
```

RoboTwin 客户端（单任务或 50 tasks）：

```bash
cd "$STREAMVGGT_ROOT"
ROBOTWIN_REPO="$ROBOTWIN_REPO" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
TASK_CONFIG=demo_clean TEST_NUM=10 \
MAX_EPISODE_STEPS=400 \
USE_CACHED_SPARSE_WAYPOINTS=1 \
SPARSE_WAYPOINT_DATASET_ROOT="$DATA_ROOT" \
EVALUATION_SEED_ROOT="$DATA_ROOT" \
SPARSE_WAYPOINT_STEP_SCALE=1.5 \
  bash src/scripts/launch_client.sh
```

也可以使用 RoboTwin 侧的编排脚本（会负责等待数据清单、启动服务、执行客户端并保存控制文件）：

```bash
cd "$ROBOTWIN_REPO"
ROBOTWIN_REPO="$ROBOTWIN_REPO" \
STREAMVGGT_ROOT="$STREAMVGGT_ROOT" \
CHECKPOINT_DIR="$CHECKPOINT_DIR" \
RESULTS_ROOT="$STREAMVGGT_ROOT/evaluation_runs/checkpoint_step_28000" \
python script/run_clean_test50_eval.py
```

固定评测约定：`checkpoint_step_28000`、50 tasks、每 task 10 demos、任务级 `MAX_EPISODE_STEPS`、`SPARSE_WAYPOINT_STEP_SCALE=1.5`。实际任务步数应以任务配置/本次运行环境为准，并将最终值写入结果元数据；不要只凭结果目录名推断配置。服务端和客户端需要在同一台可访问仿真与模型服务的机器上运行。

## 5. 消融实验清单

每个条件使用同一 checkpoint、任务集合、demo 数、seed 清单和 step limit；只改变标注缺失/噪声条件。建议结果目录命名为 `evaluation_runs/checkpoint_step_28000/<condition>/`，并保存命令、环境变量、git commit、成功率汇总和失败日志。

| 编号 | 条件 | 说明 | 建议状态 |
|---|---|---|---|
| B0 | baseline | 完整、干净 waypoint | 待运行 |
| W1 | clean waypoint | 干净 waypoint，对照 B0 的显式配置 | 待运行 |
| W1.5 | medium noise | waypoint 中等噪声 | 待运行 |
| W3 | strong noise | waypoint 强噪声 | 待运行 |
| M1 | 50% waypoint missing | 随机/预定义缺失 50% waypoint | 待运行 |
| M2-L | 单侧 waypoint missing | 左侧 waypoint 缺失 | 待运行 |
| M2-R | 单侧 waypoint missing | 右侧 waypoint 缺失 | 待运行 |
| M3 | all waypoint missing | 全部 waypoint 缺失 | 待运行 |
| C1 | head missing | head camera 缺失 | 待运行 |
| C2-L | 单侧 wrist missing | 左侧 wrist camera 缺失 | 待运行 |
| C2-R | 单侧 wrist missing | 右侧 wrist camera 缺失 | 待运行 |
| C3 | both wrist missing | 双侧 wrist camera 缺失 | 待运行 |

最少记录以下字段：condition、checkpoint、50-task 列表、每 task 的 10 个 demo/seed、任务级 step limit、waypoint 缩放、缺失/噪声生成配置、成功率、失败原因、服务端/客户端日志路径。`B0` 与 `W1` 的差异必须在实验记录中明确；若两者配置完全相同，应在报告中说明它们是重复的 sanity check。

## 6. 结果检查

```bash
python -m pytest -q robotwin-labeled/tests/test_sparse_waypoint_evaluation.py
python -m compileall -q src
```

提交前检查：

- `grep -R` 或编辑器搜索结果中不应出现机器专用绝对路径（如 `/inspire/`、`/mnt/`、`/root/`）；
- 结果目录中有 checkpoint 配置快照和运行参数；
- 50 个任务均有可追溯的 10-demo seed；
- 失败任务没有被静默丢弃，汇总文件与原始日志一致。
