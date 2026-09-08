import os
from copy import deepcopy

from easydict import EasyDict

from .va_vga_robotwin_action_only_cfg import va_vga_robotwin_action_only_cfg


va_vga_robotwin_action_only_train_cfg = EasyDict(__name__="Config: VGA robotwin action only train")
va_vga_robotwin_action_only_train_cfg.update(deepcopy(va_vga_robotwin_action_only_cfg))

va_vga_robotwin_action_only_train_cfg.dataset_type = "robotwin_lerobot"
va_vga_robotwin_action_only_train_cfg.dataset_path = "/inspire/hdd/global_user/yangdongshen-253108120197/code/robotwin-labeled/data"
va_vga_robotwin_action_only_train_cfg.use_expert_marked_rgb = True
va_vga_robotwin_action_only_train_cfg.robotwin_action_space = "joint"
va_vga_robotwin_action_only_train_cfg.joint_action_representation = "absolute"
va_vga_robotwin_action_only_train_cfg.empty_emb_path = os.path.join(
    va_vga_robotwin_action_only_train_cfg.dataset_path,
    "empty_emb.pt",
)
va_vga_robotwin_action_only_train_cfg.metric_logger = "tensorboard"
va_vga_robotwin_action_only_train_cfg.enable_wandb = True
va_vga_robotwin_action_only_train_cfg.wandb_mode = "online"
va_vga_robotwin_action_only_train_cfg.tensorboard_log_dir = None
va_vga_robotwin_action_only_train_cfg.load_worker = 2
va_vga_robotwin_action_only_train_cfg.dataset_init_worker = 0
va_vga_robotwin_action_only_train_cfg.dataset_mp_start_method = "spawn"
va_vga_robotwin_action_only_train_cfg.encode_text_in_dataloader = False
va_vga_robotwin_action_only_train_cfg.dataloader_timeout = 120
va_vga_robotwin_action_only_train_cfg.save_interval = 2000
va_vga_robotwin_action_only_train_cfg.gc_interval = 50
va_vga_robotwin_action_only_train_cfg.cfg_prob = 0.1
va_vga_robotwin_action_only_train_cfg.single_trajectory_repo_id = None

va_vga_robotwin_action_only_train_cfg.learning_rate = 1e-4
va_vga_robotwin_action_only_train_cfg.beta1 = 0.9
va_vga_robotwin_action_only_train_cfg.beta2 = 0.95
va_vga_robotwin_action_only_train_cfg.weight_decay = 0.1
va_vga_robotwin_action_only_train_cfg.warmup_steps = 10
va_vga_robotwin_action_only_train_cfg.batch_size = 12
va_vga_robotwin_action_only_train_cfg.gradient_accumulation_steps = 1
va_vga_robotwin_action_only_train_cfg.num_steps = 100000
va_vga_robotwin_action_only_train_cfg.single_task = None
