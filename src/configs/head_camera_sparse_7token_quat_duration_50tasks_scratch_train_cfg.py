"""50-task seven-token structured waypoint training from base pretrained weights.

This experiment intentionally does not load the verified 30k sparse-AWE checkpoint.
The VGA transformer is initialized from the base StreamVGGT checkpoint and the RDT
action head is initialized according to the base model construction path.
"""
from copy import deepcopy

from .head_camera_sparse_7token_quat_duration_50tasks_train_cfg import (
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg,
)

head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg = deepcopy(
    head_camera_sparse_legacy_ee_frozen_7token_50tasks_train_cfg
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.__name__ = (
    "Config: seven-token SE(3) guidance, frozen VGA, 50 RoboTwin tasks, scratch action head"
)
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.save_root = (
    "/inspire/hdd/global_user/yangdongshen-253108120197/code/StreamVGGT/"
    "train_out/head_camera_sparse_7token_quat_duration_50tasks_scratch_stage1"
)
# Explicitly disable both warm-start and optimizer-state resume. Do not inherit
# the 30k checkpoint paths from the warm-start experiment.
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.transformer_warm_start = None
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.action_head_warm_start = None
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.transformer_resume = False
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.action_head_resume = False
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.transformer_resume_from = None
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.action_head_resume_from = None
head_camera_sparse_legacy_ee_frozen_7token_50tasks_scratch_train_cfg.num_steps = 36000
