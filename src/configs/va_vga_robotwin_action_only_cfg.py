from copy import deepcopy

from easydict import EasyDict

from .va_vga_robotwin_cfg import va_vga_robotwin_cfg


va_vga_robotwin_action_only_cfg = EasyDict(__name__="Config: VGA robotwin action only")
va_vga_robotwin_action_only_cfg.update(deepcopy(va_vga_robotwin_cfg))

# Disable the ee-target subsystem entirely: no head, no condition tokens,
# no auxiliary ee-target loss, and no GT/predicted mixing.
va_vga_robotwin_action_only_cfg.enable_ee_target_module = False
va_vga_robotwin_action_only_cfg.enable_ee_target_head_eval = False
va_vga_robotwin_action_only_cfg.enable_ee_target_loss = False
va_vga_robotwin_action_only_cfg.loss_weight_ee_target = 0.0
va_vga_robotwin_action_only_cfg.use_ee_target_as_rdt_condition = False
va_vga_robotwin_action_only_cfg.ee_target_condition_gt_mix_enabled = False

if hasattr(va_vga_robotwin_action_only_cfg, "rdt_condition_tokens"):
    va_vga_robotwin_action_only_cfg.rdt_condition_tokens.use_ee_target_tokens = False
    va_vga_robotwin_action_only_cfg.rdt_condition_tokens.ee_target_layers = []

va_vga_robotwin_action_only_cfg.rdt.depth = (
    len(va_vga_robotwin_action_only_cfg.rdt_condition_tokens.image_layers)
    + len(va_vga_robotwin_action_only_cfg.rdt_condition_tokens.action_layers)
)
