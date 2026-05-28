import os

from easydict import EasyDict

from .va_vga_robotwin_cfg import va_vga_robotwin_cfg


va_vga_robotwin_train_cfg = EasyDict(__name__="Config: VGA robotwin train")
va_vga_robotwin_train_cfg.update(va_vga_robotwin_cfg)

va_vga_robotwin_train_cfg.dataset_path = "/home/yds/dataset/robotwin_labeled/adjust_bottle/demo_randomized/"
va_vga_robotwin_train_cfg.empty_emb_path = os.path.join(
    va_vga_robotwin_train_cfg.dataset_path,
    "empty_emb.pt",
)
# va_vga_robotwin_train_cfg.metric_logger = "wandb"
va_vga_robotwin_train_cfg.metric_logger = "tensorboard"
va_vga_robotwin_train_cfg.enable_wandb = True  # Backward-compatible fallback when metric_logger is unset.
va_vga_robotwin_train_cfg.wandb_mode = "online"
va_vga_robotwin_train_cfg.tensorboard_log_dir = None
va_vga_robotwin_train_cfg.load_worker = 2
va_vga_robotwin_train_cfg.dataset_init_worker = 0
va_vga_robotwin_train_cfg.dataset_mp_start_method = "spawn"
va_vga_robotwin_train_cfg.encode_text_in_dataloader = False
va_vga_robotwin_train_cfg.dataloader_timeout = 120
va_vga_robotwin_train_cfg.save_interval = 1000
va_vga_robotwin_train_cfg.gc_interval = 50
va_vga_robotwin_train_cfg.cfg_prob = 0.1
va_vga_robotwin_train_cfg.single_trajectory_repo_id = None

# Training parameters
va_vga_robotwin_train_cfg.learning_rate = 1e-4
va_vga_robotwin_train_cfg.beta1 = 0.9
va_vga_robotwin_train_cfg.beta2 = 0.95
va_vga_robotwin_train_cfg.weight_decay = 0.1
va_vga_robotwin_train_cfg.warmup_steps = 10
va_vga_robotwin_train_cfg.batch_size = 2
va_vga_robotwin_train_cfg.gradient_accumulation_steps = 1
va_vga_robotwin_train_cfg.num_steps = 100000
va_vga_robotwin_train_cfg.single_task = None
