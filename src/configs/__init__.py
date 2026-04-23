# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_vga_robotwin_cfg import va_vga_robotwin_cfg
from .va_vga_robotwin_train_cfg import va_vga_robotwin_train_cfg

VA_CONFIGS = {
    "robotwin": va_robotwin_cfg,
    "robotwin_train": va_robotwin_train_cfg,
    "vga_robotwin": va_vga_robotwin_cfg,
    "vga_robotwin_train": va_vga_robotwin_train_cfg,
}
