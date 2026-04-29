# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import math
import os
import sys
import re
import shutil
from pathlib import Path

from datetime import datetime
import wandb

import hydra
from omegaconf import OmegaConf
import pathlib

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
from safetensors import safe_open
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
# from modules.utils import (
#     load_transformer,
# )
from utils import (
    init_logger,
    logger,
    get_mesh_id,
    sample_timestep_id,
    warmup_constant_lambda,
    FlowMatchScheduler,
)
from vga.models.vga import VGA
from rdt.model import RDT
from vga.utils.lora import extract_lora_state_dict, load_lora_state_dict

from dataset import MultiLatentLeRobotDataset, MultiVGARobotwinDataset
import gc


def get_effective_num_image_views(config):
    mode = getattr(config, "multi_view_image_mode", "vertical")
    if mode == "vertical":
        return len(config.obs_cam_keys)
    if mode in {"frame", "first"}:
        return 1
    raise ValueError(f"Unsupported multi_view_image_mode `{mode}`")


def _to_plain_config(value):
    if isinstance(value, dict):
        return {str(k): _to_plain_config(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_config(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return value


CHECKPOINT_SUCCESS_MARKER = "_SUCCESS"


def _atomic_json_dump(payload, dst_path):
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp")
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, dst_path)


def _atomic_torch_save(payload, dst_path):
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, dst_path)


def _is_valid_safetensors(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False


def _atomic_safetensors_save(state_dict, dst_path):
    dst_path = Path(dst_path)
    tmp_path = dst_path.with_name(f".{dst_path.name}.tmp")
    save_file(state_dict, str(tmp_path))
    if not _is_valid_safetensors(tmp_path):
        raise RuntimeError(f"Saved invalid safetensors artifact: {tmp_path}")
    os.replace(tmp_path, dst_path)


def _is_complete_checkpoint_dir(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    marker = checkpoint_dir / CHECKPOINT_SUCCESS_MARKER
    transformer_path = checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors"
    action_head_path = checkpoint_dir / "action_head" / "diffusion_pytorch_model.safetensors"
    training_state_path = checkpoint_dir / "training_state.pt"
    return (
        marker.exists()
        and _is_valid_safetensors(transformer_path)
        and _is_valid_safetensors(action_head_path)
        and training_state_path.exists()
    )


def _find_latest_valid_checkpoint_dir(ckpt_root):
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.exists():
        return None

    pattern = re.compile(r"checkpoint_step_(\d+)$")
    candidates = []
    for p in ckpt_root.glob("checkpoint_step_*"):
        if not p.is_dir():
            continue
        m = pattern.match(p.name)
        if not m:
            continue
        candidates.append((int(m.group(1)), p))

    for _, checkpoint_dir in sorted(candidates, key=lambda x: x[0], reverse=True):
        if _is_complete_checkpoint_dir(checkpoint_dir):
            return checkpoint_dir
    return None


def _extract_modulelist_indices_from_state(state, prefix):
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)\.")
    indices = set()
    for key in state.keys():
        match = pattern.match(key)
        if match:
            indices.add(int(match.group(1)))
    return sorted(indices)


def _build_even_layer_map(src_indices, target_depth):
    if target_depth <= 0:
        raise ValueError(f"target_depth must be > 0, got {target_depth}")
    if len(src_indices) == 0:
        return {}

    src_count = len(src_indices)
    sampled_positions = [
        min((i * src_count) // target_depth, src_count - 1)
        for i in range(target_depth)
    ]
    sampled_src_indices = [src_indices[pos] for pos in sampled_positions]
    return {src_idx: dst_idx for dst_idx, src_idx in enumerate(sampled_src_indices)}


def _remap_modulelist_state_dict(state, prefix, target_depth):
    src_indices = _extract_modulelist_indices_from_state(state, prefix)
    if len(src_indices) == 0:
        return state, None

    if len(src_indices) == target_depth and src_indices == list(range(target_depth)):
        return state, src_indices

    src_to_dst = _build_even_layer_map(src_indices, target_depth)
    pattern = re.compile(rf"^{re.escape(prefix)}\.(\d+)(\..+)$")

    remapped = {}
    for key, value in state.items():
        match = pattern.match(key)
        if not match:
            remapped[key] = value
            continue

        src_idx = int(match.group(1))
        suffix = match.group(2)
        if src_idx in src_to_dst:
            dst_idx = src_to_dst[src_idx]
            remapped[f"{prefix}.{dst_idx}{suffix}"] = value

    sampled_src_indices = [src_idx for src_idx, _ in sorted(src_to_dst.items(), key=lambda x: x[1])]
    return remapped, sampled_src_indices


def _adapt_transformer_state_for_depth(state, target_depth, rank=0):
    adapted = dict(state)
    prefixes = [
        "aggregator.frame_blocks",
        "aggregator.frame_blocks_image",
        "aggregator.frame_blocks_action",
        "aggregator.global_blocks",
        "aggregator.cross_blocks",
    ]

    for prefix in prefixes:
        adapted, sampled = _remap_modulelist_state_dict(adapted, prefix, target_depth)
        if sampled is not None and rank == 0 and len(sampled) > 0:
            logger.info(
                f"Layer remap for {prefix}: sampled pretrained layers {sampled} -> target depth {target_depth}"
            )

    return adapted


def _strip_state_dict_prefixes(state):
    stripped = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "transformer.", "_orig_mod."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        stripped[new_key] = value
    return stripped


def _adapt_streamvggt_state_for_vga(state, target_depth, rank=0):
    """Map StreamVGGT checkpoint keys to VGA keys and remap depth if needed."""
    source = _strip_state_dict_prefixes(state)
    adapted = {}

    for key, value in source.items():
        if key.startswith("aggregator.frame_blocks."):
            suffix = key[len("aggregator.frame_blocks.") :]
            adapted[f"aggregator.frame_blocks_image.{suffix}"] = value
            adapted[f"aggregator.frame_blocks_action.{suffix}"] = value
            continue

        if key.startswith("aggregator.global_blocks."):
            suffix = key[len("aggregator.global_blocks.") :]
            adapted[f"aggregator.global_blocks.{suffix}"] = value
            continue

        # StreamVGGT-only heads are intentionally ignored.
        if key.startswith("point_head.") or key.startswith("track_head."):
            continue

        adapted[key] = value

    adapted = _adapt_transformer_state_for_depth(adapted, target_depth=target_depth, rank=rank)
    return adapted


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            # wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                # entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self._warned_lang_dim_mismatch = False
        self.device = torch.device(f"cuda:{config.local_rank}")
        # self.device = torch.device(f"cuda:0")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.model_arch = "vga"

        requested_arch = str(getattr(config, "model_arch", "vga")).lower()
        if requested_arch != "vga":
            raise ValueError(
                f"train_va.py now supports VGA + RDT only, got model_arch={requested_arch}"
            )

        # Load models
        logger.info("Loading models...")

        self.enable_camera_loss = bool(getattr(config, "enable_camera_loss", False))
        self.enable_depth_loss = bool(getattr(config, "enable_depth_loss", False))
        self.loss_weight_camera = float(getattr(config, "loss_weight_camera", 0.0))
        self.loss_weight_depth = float(getattr(config, "loss_weight_depth", 0.0))
        self.loss_weight_action = float(getattr(config, "loss_weight_action", 1.0))
        self.state_noise_std = float(getattr(config, "state_noise_std", 0.0))
        self.state_noise_clip = bool(getattr(config, "state_noise_clip", True))
        self.use_lora = bool(getattr(config, "use_lora", True))
        self.lora_rank = int(getattr(config, "lora_rank", 8))
        self.lora_alpha = float(getattr(config, "lora_alpha", 16.0))
        self.lora_dropout = float(getattr(config, "lora_dropout", 0.05))
        self.lora_target_modules = tuple(getattr(config, "lora_target_modules", ("qkv", "proj", "fc1", "fc2")))

        # Bypass geometry heads when corresponding losses are disabled.
        enable_camera_head = self.enable_camera_loss and self.loss_weight_camera > 0.0
        enable_depth_head = self.enable_depth_loss and self.loss_weight_depth > 0.0
        enable_geometry_heads_train = enable_camera_head or enable_depth_head

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        common_kwargs = dict(
            img_height=config.image_height,
            img_width=config.image_width,
            num_image_views=get_effective_num_image_views(self.config),
            text_embed_dim=int(getattr(config, "text_embed_dim", 4096)),
            rdt_img_cond_mode=getattr(config, "rdt_img_cond_mode", "full"),
            rdt_img_pool_size=getattr(config, "rdt_img_pool_size", 1),
            rdt_img_keep_summary_tokens=getattr(config, "rdt_img_keep_summary_tokens", False),
            aggregator_depth=int(getattr(config, "actionvggt_depth", 24)),
            window_size=1,
            chunk_size=int(getattr(config, "chunk_size", 24)),
            action_dim=int(getattr(config, "action_dim", 30)),
            image_frame_stride=int(getattr(config, "image_frame_stride", 8)),
        )
        self.transformer = VGA(
            rdt_condition_tokens=getattr(config, "rdt_condition_tokens", None),
            enable_camera_depth_heads=enable_geometry_heads_train,
            enable_camera_head=enable_camera_head,
            enable_depth_head=enable_depth_head,
            **common_kwargs,
        )
        self.transformer.config = _to_plain_config({
            "model_arch": self.model_arch,
            "img_height": config.image_height,
            "img_width": config.image_width,
            "patch_size": self.transformer.patch_size,
            "embed_dim": self.transformer.embed_dim,
            "action_dim": self.transformer.action_dim,
            "chunk_size": self.transformer.chunk_size,
            "num_image_views": self.transformer.num_image_views,
            "text_embed_dim": self.transformer.text_embed_dim,
            "rdt_img_cond_mode": self.transformer.rdt_img_cond_mode,
            "rdt_img_pool_size": self.transformer.rdt_img_pool_size,
            "rdt_img_keep_summary_tokens": self.transformer.rdt_img_keep_summary_tokens,
            "aggregator_depth": self.transformer.aggregator.depth,
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "lora_target_modules": list(self.lora_target_modules),
        })
        logger.info(f"All model parameters: {sum(p.numel() for p in self.transformer.parameters())}")

        self.transformer.to(self.device)

        rdt_config = config.rdt
        num_input_frames = 1
        effective_num_image_views = get_effective_num_image_views(self.config)
        patch_h = self.transformer.img_height // self.transformer.patch_size
        patch_w = self.transformer.img_width // self.transformer.patch_size
        pooled_patch_h = max(1, math.ceil(patch_h / self.transformer.rdt_img_pool_size))
        pooled_patch_w = max(1, math.ceil(patch_w / self.transformer.rdt_img_pool_size))
        if self.transformer.rdt_img_cond_mode == "pool":
            pooled_tokens_per_view = pooled_patch_h * pooled_patch_w
            img_tokens_per_frame = pooled_tokens_per_view * effective_num_image_views
            if self.transformer.rdt_img_keep_summary_tokens:
                img_tokens_per_frame += effective_num_image_views
                rdt_img_pos_emb_config = [("image", num_input_frames * img_tokens_per_frame)]
            else:
                rdt_img_pos_emb_config = [
                    ("image", (num_input_frames * effective_num_image_views, pooled_patch_h, pooled_patch_w))
                ]
        else:
            img_tokens_per_frame = patch_h * patch_w * effective_num_image_views
            rdt_img_pos_emb_config = [
                ("image", (num_input_frames * effective_num_image_views, patch_h, patch_w))
            ]
        act_tokens_per_frame = self.config.image_frame_stride
        rdt_horizon = self.config.chunk_size
        rdt_x_pos_emb_config = [("act", rdt_horizon + self.config.rdt.num_register_tokens)]
        rdt_act_pos_emb_config = [("action", (num_input_frames, act_tokens_per_frame))]

        self.action_head = RDT(
            horizon=rdt_horizon,
            output_size=self.transformer.action_dim,
            config=rdt_config,
            x_pos_emb_config=rdt_x_pos_emb_config,
            lang_pos_emb_config=None,
            max_lang_len=0,
            img_pos_emb_config=rdt_img_pos_emb_config,
            max_img_len=num_input_frames * img_tokens_per_frame,
            act_pos_emb_config=rdt_act_pos_emb_config,
            max_act_len=num_input_frames * act_tokens_per_frame,
            dtype=self.dtype,
        )
        self.action_head.config = _to_plain_config({
            "horizon": rdt_horizon,
            "output_size": self.transformer.action_dim,
            "text_embed_dim": int(getattr(config, "text_embed_dim", 4096)),
            "config": rdt_config,
            "x_pos_emb_config": rdt_x_pos_emb_config,
            "lang_pos_emb_config": None,
            "max_lang_len": 0,
            "img_pos_emb_config": rdt_img_pos_emb_config,
            "max_img_len": num_input_frames * img_tokens_per_frame,
            "act_pos_emb_config": rdt_act_pos_emb_config,
            "max_act_len": num_input_frames * act_tokens_per_frame,
            "dtype": self.dtype,
        })
        self.action_head.to(self.device)

        def _load_checkpoint_state(path):
            if str(path).endswith('.safetensors'):
                try:
                    return load_file(path, device=str(self.device))
                except Exception as exc:
                    raise RuntimeError(f"Failed to load safetensors checkpoint: {path}") from exc
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict):
                for top_key in ('state_dict', 'model_state_dict', 'model'):
                    if top_key in state and isinstance(state[top_key], dict):
                        return state[top_key]
            return state

        def _resize_pos_embed_tensor(src_pos_embed: torch.Tensor, dst_pos_embed: torch.Tensor):
            if src_pos_embed.ndim != 3 or dst_pos_embed.ndim != 3:
                return None
            if src_pos_embed.shape[-1] != dst_pos_embed.shape[-1]:
                return None

            src_len = src_pos_embed.shape[1]
            dst_len = dst_pos_embed.shape[1]
            if src_len == dst_len:
                return src_pos_embed

            src_prefix = 1
            dst_prefix = 1
            src_grid_tokens = src_len - src_prefix
            dst_grid_tokens = dst_len - dst_prefix
            src_hw = int(math.sqrt(src_grid_tokens))
            dst_hw = int(math.sqrt(dst_grid_tokens))

            if src_hw * src_hw != src_grid_tokens or dst_hw * dst_hw != dst_grid_tokens:
                return None

            src_prefix_tokens = src_pos_embed[:, :src_prefix]
            src_grid = src_pos_embed[:, src_prefix:]
            src_grid = src_grid.reshape(1, src_hw, src_hw, -1).permute(0, 3, 1, 2)
            src_grid = F.interpolate(src_grid, size=(dst_hw, dst_hw), mode="bicubic", align_corners=False)
            src_grid = src_grid.permute(0, 2, 3, 1).reshape(1, dst_hw * dst_hw, -1)

            return torch.cat([src_prefix_tokens, src_grid], dim=1)

        def _adapt_transformer_state_for_resolution(model: torch.nn.Module, state: dict):
            if not isinstance(state, dict):
                return state

            adapted = dict(state)
            model_state = model.state_dict()
            pos_key = "aggregator.patch_embed.pos_embed"

            if pos_key in adapted and pos_key in model_state:
                src_pos = adapted[pos_key]
                dst_pos = model_state[pos_key]
                if src_pos.shape != dst_pos.shape:
                    resized = _resize_pos_embed_tensor(src_pos, dst_pos)
                    if resized is None:
                        logger.warning(
                            f"Skip loading {pos_key}: checkpoint shape {tuple(src_pos.shape)} "
                            f"!= model shape {tuple(dst_pos.shape)} and resize is not applicable"
                        )
                        adapted.pop(pos_key, None)
                    else:
                        adapted[pos_key] = resized.to(dtype=dst_pos.dtype, device=dst_pos.device)
                        logger.info(
                            f"Resized {pos_key} from {tuple(src_pos.shape)} to {tuple(dst_pos.shape)}"
                        )

            return adapted

        streamvggt_pretrained = getattr(config, "streamvggt_pretrained", None)
        if streamvggt_pretrained is None:
            streamvggt_pretrained = str(Path(__file__).resolve().parent.parent / "ckpt" / "checkpoints.pth")
        streamvggt_pretrained = Path(streamvggt_pretrained)
        if not streamvggt_pretrained.exists():
            raise FileNotFoundError(f"StreamVGGT checkpoint not found: {streamvggt_pretrained}")

        transformer_resume_from = getattr(config, "transformer_resume_from", None)
        action_head_resume_from = getattr(config, "action_head_resume_from", None)
        self._resume_checkpoint_dir = None

        logger.info(f"Initializing VGA from StreamVGGT checkpoint: {streamvggt_pretrained}")
        transformer_state = _load_checkpoint_state(streamvggt_pretrained)
        transformer_state = _adapt_streamvggt_state_for_vga(
            transformer_state,
            target_depth=self.transformer.aggregator.depth,
            rank=self.config.rank,
        )
        transformer_state = _adapt_transformer_state_for_resolution(self.transformer, transformer_state)
        logger.info(self.transformer.load_state_dict(transformer_state, strict=False))

        if self.use_lora:
            lora_modules = self.transformer.enable_lora(
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
                target_modules=self.lora_target_modules,
            )
            self.transformer.prepare_lora_training()
            if self.config.rank == 0:
                logger.info(
                    f"Enabled LoRA on VGA backbone: rank={self.lora_rank}, alpha={self.lora_alpha}, "
                    f"dropout={self.lora_dropout}, modules={len(lora_modules)}"
                )

        if getattr(config, "transformer_resume", False) and transformer_resume_from:
            transformer_resume_from = Path(transformer_resume_from)
            self._resume_checkpoint_dir = transformer_resume_from.parent.parent
            if not _is_complete_checkpoint_dir(self._resume_checkpoint_dir):
                fallback_dir = _find_latest_valid_checkpoint_dir(self._resume_checkpoint_dir.parent)
                if fallback_dir is None:
                    raise RuntimeError(
                        f"Requested resume checkpoint is incomplete/corrupt and no valid fallback found under "
                        f"{self._resume_checkpoint_dir.parent}: {self._resume_checkpoint_dir}"
                    )
                logger.warning(
                    f"Requested resume checkpoint is incomplete/corrupt: {self._resume_checkpoint_dir}. "
                    f"Falling back to latest valid checkpoint: {fallback_dir}"
                )
                self._resume_checkpoint_dir = fallback_dir
                transformer_resume_from = self._resume_checkpoint_dir / "transformer" / "diffusion_pytorch_model.safetensors"
            logger.info(f"Resuming VGA transformer from {transformer_resume_from}")
            transformer_state = _load_checkpoint_state(transformer_resume_from)
            transformer_state = _adapt_transformer_state_for_resolution(self.transformer, transformer_state)
            logger.info(self.transformer.load_state_dict(transformer_state, strict=False))

            lora_state_path = self._resume_checkpoint_dir / "transformer" / "lora_weights.safetensors"
            if lora_state_path.exists():
                logger.info(f"Loading LoRA delta from {lora_state_path}")
                lora_state = _load_checkpoint_state(lora_state_path)
                logger.info(load_lora_state_dict(self.transformer, lora_state, strict=False))
        elif self.use_lora and self.config.rank == 0:
            logger.info("No transformer resume checkpoint configured; starting LoRA from pretrained backbone.")

        if self.config.rank == 0:
            logger.info("Initializing RDT action head from scratch (no pretrained load).")

        if config.gradient_checkpointing:
            logger.info("Enabling activation checkpointing on transformer and action head...")
            apply_ac(self.transformer)
            apply_ac(self.action_head)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.action_head = _configure_model(
            model=self.action_head,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )

        self.transformer.train()
        self.transformer.requires_grad_(True)

        self.action_head.train()
        self.action_head.requires_grad_(True)

        frozen_params = 0
        total_params = 0
        frozen_param_names = []

        for name, param in self.transformer.named_parameters():
            total_params += param.numel()
            if self.use_lora:
                should_train = ("lora_" in name) or name.endswith("action_query_tokens")
                param.requires_grad = should_train
            else:
                param.requires_grad = True

        if self.use_lora:
            for name, p in self.transformer.named_parameters():
                if not p.requires_grad:
                    frozen_params += p.numel()
                    frozen_param_names.append(name)
        else:
            logger.info("LoRA disabled: training VGA backbone normally.")

        logger.info(
            f"Frozen {frozen_params:,} parameters out of {total_params:,} total parameters. ({frozen_params / total_params:.2%})")
        logger.info(
            f"Trainable parameters: {total_params - frozen_params:,} ({(total_params - frozen_params) / total_params:.2%})")
        if frozen_param_names:
            logger.info(
                f"Example frozen parameters: {', '.join(frozen_param_names[:5])}{'...' if len(frozen_param_names) > 5 else ''}")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad]
            + [p for p in self.action_head.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        dataset_type = str(getattr(config, "dataset_type", "robotwin")).lower()
        if dataset_type == "vga_robotwin":
            train_dataset = MultiVGARobotwinDataset(config=config)
        else:
            train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        if self._resume_checkpoint_dir is not None:
            self.save_dir = self._resume_checkpoint_dir.parent
            logger.info(f"Resumed run will continue writing checkpoints under existing root: {self.save_dir}")
        else:
            self.save_dir = Path(config.save_root) / datetime.now().strftime("train_log_%Y%m%d_%H%M%S") / "ckpt" # Add timestamp YYMMDD_HHMMSS to save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        if getattr(config, "transformer_resume", False) and self._resume_checkpoint_dir is not None:
            self._load_training_state(self._resume_checkpoint_dir)
            action_head_resume_from = action_head_resume_from or (self._resume_checkpoint_dir / "action_head" / "diffusion_pytorch_model.safetensors")
        if getattr(config, "action_head_resume", False) and action_head_resume_from:
            action_head_resume_from = Path(action_head_resume_from)
            logger.info(f"Resuming action head from {action_head_resume_from}")
            action_head_state = _load_checkpoint_state(action_head_resume_from)
            logger.info(self.action_head.load_state_dict(action_head_state, strict=False))
    
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=B, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=0)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=B,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=0)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[:, None].repeat(1, F),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[:, None].repeat(1, F),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare model input dict from fixed-size dataset outputs."""
        chunk_size = getattr(self.config, 'chunk_size', None)

        if 'images' not in batch_dict:
            raise ValueError("batch_dict must include raw 'images' for VGA training")
        if 'actions' not in batch_dict:
            raise ValueError("batch_dict must include 'actions' for VGA training")
        if 'action_chunk' not in batch_dict:
            raise ValueError("batch_dict must include 'action_chunk' for VGA training")
        if 'pred_frame_idx' not in batch_dict:
            raise ValueError("batch_dict must include 'pred_frame_idx' for VGA training")

        images = batch_dict['images']  # [B, C_image, F, H, W]
        actions = batch_dict['actions']  # [B, C_action, F, N, 1]
        text_emb = batch_dict.get('text_emb', None)
        image_mask = batch_dict.get('images_mask', torch.ones_like(images, dtype=torch.bool))
        action_mask = batch_dict.get('actions_mask', torch.ones_like(actions, dtype=torch.bool))

        B, _, F, H, W = images.shape
        B_action, _, F_action, N, _ = actions.shape
        if B != B_action or F != F_action:
            raise ValueError(
                f"images/actions shape mismatch: images={tuple(images.shape)}, actions={tuple(actions.shape)}"
            )


        # Build grid_id for image tokens using 3D mesh (F, H//p, W//p)
        patch_f, patch_h, patch_w = self.patch_size
        image_grid_id = get_mesh_id(
            F // patch_f,
            H // patch_h,
            W // patch_w,
            t=0,
            f_w=1,
            f_shift=0,
            action=False,
        ).to(self.device)
        image_grid_id = image_grid_id[None].repeat(B, 1, 1)

        image_dict = dict(
            images=images,
            grid_id=image_grid_id,
            text_emb=batch_dict.get('text_emb', None),
            images_mask=image_mask,
        )

        # Replace action grid_id to align with action token RoPE (one token per frame)
        action_grid_id = get_mesh_id(
            F,
            N,
            1,
            t=1,
            f_w=1,
            f_shift=0,
            action=True,
        ).to(self.device)
        action_grid_id = action_grid_id[None].repeat(B, 1, 1)

        action_dict = dict(
            actions=actions,
            grid_id=action_grid_id,
            text_emb=batch_dict.get('text_emb', None),
            action_mask=action_mask,
            actions_mask=action_mask,
        )

        # Build noised action chunks from dataset-provided chunk targets.
        action_chunk = batch_dict['action_chunk']  # [B, C_action, chunk_size]
        if action_chunk.shape[-1] != chunk_size:
            raise ValueError(
                f"action_chunk length mismatch: got {action_chunk.shape[-1]}, expected {chunk_size}"
            )
        action_chunk = rearrange(action_chunk, 'b c f -> b c f 1 1')  # [B, C_action, chunk_size, 1, 1]
        action_chunk_dict = self._add_noise(
            action_chunk,
            train_scheduler=self.train_scheduler_action,
            action_mask=None,
            action_mode=True,
        )
        action_chunk_dict["pred_frame_idx"] = batch_dict['pred_frame_idx'].long()
        action_chunk_dict['targets'] = rearrange(action_chunk_dict['targets'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]
        action_chunk_dict['noisy_latents'] = rearrange(action_chunk_dict['noisy_latents'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]
        action_chunk_dict['latent'] = rearrange(action_chunk_dict['latent'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]

        # Keep raw language conditioning for learned projection before RDT.
        if text_emb is not None:
            if text_emb.ndim == 2:
                text_emb = text_emb.unsqueeze(1)
            elif text_emb.ndim != 3:
                raise ValueError(
                    f"text_emb must have shape [B, D] or [B, L, D], got {tuple(text_emb.shape)}"
                )
        
        input_dict = {
            'image_dict': image_dict,
            'action_dict': action_dict,
            'pred_action_chunk_dict': action_chunk_dict,
            'lang_c': text_emb,
            'chunk_size': chunk_size,
        }
        if 'state' in batch_dict:
            state = batch_dict['state']
            if state.ndim == 2:
                input_dict['state_c'] = state.unsqueeze(1)  # [B, 1, C_action]
            elif state.ndim == 3 and state.shape[1] == 1:
                input_dict['state_c'] = state
            else:
                raise ValueError(f"state must be [B, C] or [B, 1, C], got {tuple(state.shape)}")
        return input_dict

    def convert_input_format(self, input_dict):
        """Move tensors inside nested dicts to the target device."""
        def to_device(obj):
            if torch.is_tensor(obj):
                return obj.to(self.device)
            if isinstance(obj, dict):
                return {k: to_device(v) for k, v in obj.items()}
            return obj

        def to_dtype(obj):
            if torch.is_tensor(obj) and obj.dtype in [torch.float32, torch.float16, torch.float, torch.bfloat16]:
                return obj.to(self.dtype)
            if isinstance(obj, dict):
                return {k: to_dtype(v) for k, v in obj.items()}
            return obj

        return to_dtype(to_device(input_dict))

    def compute_loss(self, input_dict, pred):
        action_pred = pred
        action_pred = rearrange(action_pred, 'b f c -> b c f')
        Bn, Fn = input_dict['pred_action_chunk_dict']['timesteps'].shape
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['pred_action_chunk_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['pred_action_chunk_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :]
        # action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 1)  # (B, C, F) -> (B, F, C)
        # action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, C) -> (B*F, C)
        # action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        # action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        # action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return action_loss_per_frame.mean() / self.gradient_accumulation_steps

    def _compute_camera_depth_losses(self, batch, model_output):
        zero = torch.zeros((), device=self.device, dtype=torch.float32)
        geometry = getattr(model_output, "geometry", None)
        if not isinstance(geometry, dict):
            return zero, zero

        camera_loss = zero
        depth_loss = zero

        camera_pred = geometry.get("camera_pose", None)
        camera_gt = batch.get("camera_pose_gt", None)
        if camera_pred is not None and camera_gt is not None:
            camera_gt = camera_gt.to(device=camera_pred.device, dtype=camera_pred.dtype)
            camera_loss = F.smooth_l1_loss(camera_pred, camera_gt, reduction="mean")

        depth_pred = geometry.get("depth", None)
        depth_conf = geometry.get("depth_conf", None)
        depth_gt = batch.get("depth_gt", None)
        if depth_pred is not None and depth_conf is not None and depth_gt is not None:
            depth_gt = depth_gt.to(device=depth_pred.device, dtype=depth_pred.dtype)
            valid_mask = batch.get("depth_valid_mask", None)
            if valid_mask is not None:
                valid_mask = valid_mask.to(device=depth_pred.device, dtype=depth_pred.dtype)
            else:
                valid_mask = torch.ones_like(depth_pred)

            depth_err = torch.abs(depth_pred - depth_gt)
            aleatoric = depth_err * torch.exp(-depth_conf) + depth_conf
            aleatoric = (aleatoric * valid_mask).sum() / (valid_mask.sum() + 1e-6)

            grad_weight = float(getattr(self.config, "depth_loss_grad_weight", 0.1))
            if depth_pred.shape[-2] > 1 and depth_pred.shape[-1] > 1:
                pred_dx = depth_pred[..., :, 1:] - depth_pred[..., :, :-1]
                gt_dx = depth_gt[..., :, 1:] - depth_gt[..., :, :-1]
                pred_dy = depth_pred[..., 1:, :] - depth_pred[..., :-1, :]
                gt_dy = depth_gt[..., 1:, :] - depth_gt[..., :-1, :]
                grad_loss = F.l1_loss(pred_dx, gt_dx, reduction="mean") + F.l1_loss(pred_dy, gt_dy, reduction="mean")
            else:
                grad_loss = torch.zeros((), device=depth_pred.device, dtype=depth_pred.dtype)

            depth_loss = aleatoric + grad_weight * grad_loss

        return camera_loss, depth_loss

    def train_epoch(self):
        self.transformer.train()

        # Use manual progress bar control to only update on optimizer steps
        progress_bar = tqdm(
            total=len(self.train_loader),
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_camera_losses = []
        accumulated_depth_losses = []

        for batch_idx, batch in enumerate(self.train_loader):
            batch = self.convert_input_format(batch)

            input_dict = self._prepare_input_dict(batch)

            should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader)
            
            if not should_sync:
                self.transformer.set_requires_gradient_sync(False)
            else:
                self.transformer.set_requires_gradient_sync(True)

            output = self.transformer(input_dict)
            rdt_conds = output.ress

            action_chunk = input_dict['pred_action_chunk_dict']['noisy_latents']  # [B, C_action, chunk_size]
            action_chunk = action_chunk.permute(0, 2, 1) # [B, chunk_size, C_action]
            timesteps = input_dict['pred_action_chunk_dict']['timesteps'][:, 0]
            state_c = input_dict.get('state_c', None)
            if state_c is None:
                state_c = input_dict['pred_action_chunk_dict']['latent'][:, :, 0:1]  # [B, C_latent, 1]
                state_c = state_c.permute(0, 2, 1)  # [B, 1, C_latent]
            if self.state_noise_std > 0.0:
                state_c = state_c + torch.randn_like(state_c) * self.state_noise_std
                if self.state_noise_clip:
                    state_c = state_c.clamp(-1.0, 1.0)

            action_pred = self.action_head(
                action_chunk,
                timesteps,
                lang_c=rdt_conds['rdt_lang_c'],
                img_c=rdt_conds['rdt_img_c'],
                act_c=rdt_conds['rdt_act_c'],
                state_c=state_c,
                embed_input=True,
                decode_output=True,
            )

            action_loss = self.compute_loss(input_dict, action_pred)
            if self.model_arch == "vga":
                if self.enable_camera_loss or self.enable_depth_loss:
                    camera_loss, depth_loss = self._compute_camera_depth_losses(batch, output)
                else:
                    zero = torch.zeros((), device=self.device, dtype=torch.float32)
                    camera_loss, depth_loss = zero, zero

                loss = (
                    self.loss_weight_action * action_loss
                    + self.loss_weight_camera * camera_loss
                    + self.loss_weight_depth * depth_loss
                )
                accumulated_camera_losses.append((camera_loss / self.gradient_accumulation_steps).detach())
                accumulated_depth_losses.append((depth_loss / self.gradient_accumulation_steps).detach())
            else:
                loss = action_loss

            loss.backward()

            # Accumulate losses for logging
            # accumulated_latent_losses.append(latent_loss.detach())
            accumulated_action_losses.append(action_loss.detach())

            # Only update weights after accumulating gradients
            if should_sync:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in self.transformer.parameters() if p.requires_grad
                    ] + [
                        p for p in self.action_head.parameters() if p.requires_grad
                    ],
                    2.0,
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                if len(accumulated_latent_losses) > 0:
                    latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                    max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                else:
                    latent_loss_show = 0.0
                    max_latent_loss_show = 0.0

                if len(accumulated_camera_losses) > 0:
                    camera_loss_show = dist_mean(torch.stack(accumulated_camera_losses).sum()).detach().cpu().item()
                else:
                    camera_loss_show = 0.0
                if len(accumulated_depth_losses) > 0:
                    depth_loss_show = dist_mean(torch.stack(accumulated_depth_losses).sum()).detach().cpu().item()
                else:
                    depth_loss_show = 0.0

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_camera_losses = []
                accumulated_depth_losses = []

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    # Manually increment counter, set_postfix will refresh automatically
                    progress_bar.n += self.gradient_accumulation_steps
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'camera_loss': f'{camera_loss_show:.4f}',
                        'depth_loss': f'{depth_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_avg_camera_loss': camera_loss_show,
                            'loss_metrics/global_avg_depth_loss': depth_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                self.step += 1
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        checkpoint_dir = None
        checkpoint_tmp_dir = None
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            action_head_state_dict = get_model_state_dict(
                self.action_head,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            action_head_state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in action_head_state_dict.items()}
            # Optimizer contains params from both modules; provide both for correct FQN<->param-id mapping.
            optim_owner = torch.nn.ModuleDict({
                "transformer": self.transformer,
                "action_head": self.action_head,
            })
            optim_state = get_optimizer_state_dict(
                optim_owner,
                self.optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_tmp_dir = self.save_dir / f".checkpoint_step_{self.step}.tmp"

                if checkpoint_tmp_dir.exists():
                    shutil.rmtree(checkpoint_tmp_dir)
                checkpoint_tmp_dir.mkdir(parents=True, exist_ok=True)

                if checkpoint_dir.exists() and _is_complete_checkpoint_dir(checkpoint_dir):
                    logger.warning(f"Checkpoint already complete, skipping save: {checkpoint_dir}")
                    if dist.is_initialized():
                        dist.barrier()
                    return

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_tmp_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                _atomic_safetensors_save(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                _atomic_json_dump(config_dict, config_file)

                # Save action head in the same format as pretrained model
                action_head_dir = checkpoint_tmp_dir / "action_head"
                action_head_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving action head to {action_head_dir}")

                action_head_file = action_head_dir / "diffusion_pytorch_model.safetensors"
                _atomic_safetensors_save(action_head_state_dict_bf16, action_head_file)

                action_head_config_file = action_head_dir / "config.json"
                action_head_config_dict = dict(self.action_head.config)
                action_head_config_dict.pop('_name_or_path', None)
                _atomic_json_dump(action_head_config_dict, action_head_config_file)

                lora_state_dict = {
                    key: value.detach().to(dtype=torch.bfloat16, device="cpu").contiguous().clone()
                    for key, value in state_dict.items()
                    if "lora_" in key or key.endswith("action_query_tokens")
                }
                if lora_state_dict:
                    lora_file = transformer_dir / "lora_weights.safetensors"
                    logger.info(f"Saving LoRA delta to {lora_file}")
                    _atomic_safetensors_save(lora_state_dict, lora_file)

                plain_training_config = _to_plain_config(dict(vars(self.config)))
                training_config_path = checkpoint_tmp_dir / "training_config.json"
                logger.info(f"Saving training config to {training_config_path}")
                _atomic_json_dump(plain_training_config, training_config_path)

                training_state_path = checkpoint_tmp_dir / "training_state.pt"
                logger.info(f"Saving training state to {training_state_path}")
                _atomic_torch_save({
                    'step': self.step,
                    'optimizer_state_dict': optim_state,
                    'config': plain_training_config,
                }, training_state_path)

                _atomic_json_dump(
                    {
                        "step": int(self.step),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    },
                    checkpoint_tmp_dir / CHECKPOINT_SUCCESS_MARKER,
                )

                if checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir)
                os.replace(checkpoint_tmp_dir, checkpoint_dir)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
                if checkpoint_tmp_dir is not None and Path(checkpoint_tmp_dir).exists():
                    shutil.rmtree(checkpoint_tmp_dir, ignore_errors=True)
            # Ensure all processes stay synchronized even on error
            if dist.is_initialized():
                dist.barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        try:
            training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
        except Exception as exc:
            raise RuntimeError(f"Failed to load training_state.pt from {training_state_path}") from exc

        # All ranks load optimizer state (required for FSDP)
        optim_owner = torch.nn.ModuleDict({
            "transformer": self.transformer,
            "action_head": self.action_head,
        })
        set_optimizer_state_dict(
            optim_owner,
            self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")

        while self.step < self.config.num_steps:
            logger.info(f"Starting epoch at step {self.step}...")
            self.train_epoch()
            if dist.is_initialized():
                dist.barrier()

        logger.info("Training completed!")


def run(args): 
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if args.single_task is not None:
        config.single_task = args.single_task

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        if getattr(config, "single_task", None):
            logger.info(f"Single-task training enabled: {config.single_task}")

    trainer = Trainer(config)
    trainer.train()

def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--single-task",
        "--task_name",
        dest="single_task",
        type=str,
        default=None,
        help="Train on a single task only (same semantics as evaluation task_name)",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
