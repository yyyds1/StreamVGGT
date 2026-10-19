# ActionVGGT

ActionVGGT combines the StreamVGGT visual backbone with an RDT action head for RoboTwin sparse-waypoint control. This repository contains training, serving, evaluation, and ablation entry points.

## Repositories and branches

- ActionVGGT: https://github.com/yyyds1/StreamVGGT.git, branch `head-camera-temporal`
- RoboTwin labeled environment: https://github.com/yyyds1/robotwin-labeled.git, branch `feature/sparse-se3-guidance-data`
- ModelScope model repository: `yyyyds139/ActionVGGT`
- Training dataset: `yyyyds139/robotwin-labeled-awe-train`
- Evaluation dataset: `yyyyds139/robotwin-labeled-awe-eval`

## Installation

```bash
git clone --branch head-camera-temporal https://github.com/yyyds1/StreamVGGT.git
git clone --branch feature/sparse-se3-guidance-data git@github.com:yyyds1/robotwin-labeled.git
cd StreamVGGT
conda create -n ActionVGGT python=3.11 cmake=3.14.0 -y
conda activate ActionVGGT
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

Install RoboTwin dependencies using `robotwin-labeled/README.md` and the official RoboTwin installation guide.

## Download data and checkpoints

Install ModelScope CLI/API in the environment used for downloads. The repository names below are the source of truth; no Hugging Face or machine-specific paths are required.

```bash
modelscope download --dataset yyyyds139/robotwin-labeled-awe-eval --local_dir "$ROBOTWIN_ROOT/data_clean_test50"
# Optional training data
modelscope download --dataset yyyyds139/robotwin-labeled-awe-train --local_dir "$ROBOTWIN_ROOT/data"
# Original StreamVGGT and frozen DINOv2 assets
modelscope download --model yyyyds139/ActionVGGT --include 'original/**' --local_dir "$STREAMVGGT_ROOT/ckpt"
# ActionVGGT experiment checkpoint
modelscope download --model yyyyds139/ActionVGGT --include 'experiments/checkpoint_step_28000/**' --local_dir "$STREAMVGGT_ROOT"
```

Recommended layout:

```text
workdir/
├── StreamVGGT/
│   ├── ckpt/original/streamvggt/checkpoints.pth
│   ├── ckpt/original/dinov2-small/
│   ├── experiments/checkpoint_step_28000/
│   └── src/
└── robotwin-labeled/
    ├── data/                 # optional training set
    └── data_clean_test50/    # evaluation set
```

Set paths at runtime:

```bash
export STREAMVGGT_ROOT="$PWD/StreamVGGT"
export ROBOTWIN_ROOT="$PWD/robotwin-labeled"
export DATA_ROOT="$ROBOTWIN_ROOT/data"
export EVAL_DATA_ROOT="$ROBOTWIN_ROOT/data_clean_test50"
export CHECKPOINT_DIR="$STREAMVGGT_ROOT/experiments/checkpoint_step_28000"
```

## Training

The current training entry points are under `src/scripts/`. Configure at least `NGPU`, `DATA_ROOT`, `STREAMVGGT_ROOT`, the training config, and an output directory. Example:

```bash
cd "$STREAMVGGT_ROOT"
NGPU=8 DATA_ROOT="$DATA_ROOT" SAVE_ROOT="$STREAMVGGT_ROOT/train_out" \
  bash src/scripts/run_head_wrist_dinov2_noisy_lora_bs12_rank256_8gpu_100k.sh
```

Training outputs should contain `checkpoint_step_*`, `training_config.json`, `transformer/`, and `action_head/`.

## Evaluation

Start the model server and then the RoboTwin client on the same machine:

```bash
cd "$STREAMVGGT_ROOT"
CHECKPOINT_DIR="$CHECKPOINT_DIR" EVAL_CONFIG="$CHECKPOINT_DIR/training_config.json" \
  bash src/scripts/launch_server.sh

ROBOTWIN_REPO="$ROBOTWIN_ROOT" CHECKPOINT_DIR="$CHECKPOINT_DIR" \
TEST_NUM=10 TASK_CONFIG=demo_clean SPARSE_WAYPOINT_STEP_SCALE=1.5 \
  bash src/scripts/launch_client.sh
```

The baseline protocol is checkpoint `checkpoint_step_28000`, 50 tasks, 10 demos per task, wrist camera input, noisy waypoint input, task-level episode limits, and waypoint speed scale `1.5`.

## Ablation experiments

Each condition has a JSON profile and a shell entry point under `configs/ablation_profiles/` and `src/scripts/ablation/`. For example:

```bash
ABLATION_PROFILE=W1_clean_waypoint bash src/scripts/ablation/run_W1_clean_waypoint.sh
ABLATION_PROFILE=C3_both_wrist_missing bash src/scripts/ablation/run_C3_both_wrist_missing.sh
```

Record the command, environment variables, git commit, checkpoint, seed, task list, and summary TSV for every run.

The full Chinese collaboration guide is [docs/中文实验协作指南.md](docs/中文实验协作指南.md).
