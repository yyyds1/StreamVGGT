# ActionVGGT + RoboTwin Collaboration Handbook

The maintained collaboration instructions are in [中文实验协作指南.md](中文实验协作指南.md). This short document keeps the repository-level entry points discoverable and avoids machine-specific paths.

## Source repositories

- ActionVGGT: https://github.com/yyyds1/StreamVGGT.git, branch `head-camera-temporal`
- RoboTwin labeled environment: https://github.com/yyyds1/robotwin-labeled.git, branch `feature/sparse-se3-guidance-data`

## ModelScope assets

- Training dataset: `yyyyds139/robotwin-labeled-awe-train`
- Evaluation dataset: `yyyyds139/robotwin-labeled-awe-eval`
- Original StreamVGGT/DINOv2 and ActionVGGT checkpoints: `yyyyds139/ActionVGGT`

Use ModelScope download commands from the Chinese collaboration guide. Do not use legacy Hugging Face or machine-local checkpoint paths.

## Main entry points

```bash
# model server
bash src/scripts/launch_server.sh

# RoboTwin evaluation client
bash src/scripts/launch_client.sh

# ablation example
bash src/scripts/ablation/run_W1_clean_waypoint.sh
```

All paths should be supplied with environment variables such as `STREAMVGGT_ROOT`, `ROBOTWIN_ROOT`, `DATA_ROOT`, `EVAL_DATA_ROOT`, and `CHECKPOINT_DIR`.
