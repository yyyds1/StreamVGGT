"""Ten-task dataset variant of the validated frozen sparse Stage 1 config."""
from copy import deepcopy

from .head_camera_sparse_legacy_ee_frozen_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_train_cfg,
)

head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_train_cfg
)
head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg.__name__ = (
    "Config: sparse legacy EE frozen VGA, 10 RoboTwin tasks"
)
head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg.dataset_path = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data_sparse_awe"
)
head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg.robotwin_lerobot_repo_subdir = (
    "demo_randomized/lerobot"
)
head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg.single_task = (
    "adjust_bottle,beat_block_hammer,blocks_ranking_rgb,blocks_ranking_size,"
    "click_alarmclock,click_bell,dump_bin_bigbin,grab_roller,handover_block,handover_mic"
)
head_camera_sparse_legacy_ee_frozen_10tasks_train_cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_legacy_ee_frozen_sparse_rgb_bs12_10tasks_stage1"
)
