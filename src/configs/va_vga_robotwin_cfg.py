from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_vga_robotwin_cfg = EasyDict(__name__="Config: VGA robotwin")
va_vga_robotwin_cfg.update(va_robotwin_cfg)

va_vga_robotwin_cfg.model_arch = "vga"
va_vga_robotwin_cfg.dataset_type = "robotwin"
va_vga_robotwin_cfg.streamvggt_pretrained = "/home/yds/code/StreamVGGT/ckpt/checkpoints.pth"

# VGA training script no longer uses ActionVGGT/RDT pretrained-resume switches.
va_vga_robotwin_cfg.transformer_resume = False
va_vga_robotwin_cfg.transformer_resume_from = None
va_vga_robotwin_cfg.transformer_pretrained = None
va_vga_robotwin_cfg.action_head_resume = False
va_vga_robotwin_cfg.action_head_resume_from = None
va_vga_robotwin_cfg.action_head_pretrained = None

# LoRA settings for the pretrained VGA backbone.
va_vga_robotwin_cfg.use_lora = True
va_vga_robotwin_cfg.lora_rank = 8
va_vga_robotwin_cfg.lora_alpha = 16.0
va_vga_robotwin_cfg.lora_dropout = 0.05
va_vga_robotwin_cfg.lora_target_modules = ("qkv", "proj", "fc1", "fc2")

# VGA heads: enabled during train, bypassed during eval for efficiency.
va_vga_robotwin_cfg.enable_geometry_heads_train = False
va_vga_robotwin_cfg.enable_geometry_heads_eval = False

# Loss toggles (camera/depth code is kept and can be re-enabled anytime).
va_vga_robotwin_cfg.enable_camera_loss = False
va_vga_robotwin_cfg.enable_depth_loss = False

# Default to action-loss-only training on RobotWin.
va_vga_robotwin_cfg.loss_weight_camera = 0.0
va_vga_robotwin_cfg.loss_weight_depth = 0.0
va_vga_robotwin_cfg.loss_weight_action = 1.0

# Depth loss hyper-parameters.
va_vga_robotwin_cfg.depth_loss_grad_weight = 0.1

# RDT condition token composition from VGA backbone outputs.
va_vga_robotwin_cfg.rdt_condition_tokens = EasyDict()
va_vga_robotwin_cfg.rdt_condition_tokens.use_action_queries = True
va_vga_robotwin_cfg.rdt_condition_tokens.use_image_tokens = True
va_vga_robotwin_cfg.rdt_condition_tokens.use_language_tokens = True
