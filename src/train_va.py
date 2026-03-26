# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import math
import os
import sys
import re
from pathlib import Path
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
from actionvggt.models.actionvggt import ActionVGGT
from rdt.model import RDT

from dataset import MultiLatentLeRobotDataset
import gc


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
        self.device = torch.device(f"cuda:{config.local_rank}")
        # self.device = torch.device(f"cuda:0")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        self.transformer = ActionVGGT(
            img_height=config.image_height,
            img_width=config.image_width,
        )
        logger.info(f"All model parameters: {sum(p.numel() for p in self.transformer.parameters())}")

        self.transformer.to(self.device)

        rdt_config = config.rdt
        patch_h = self.transformer.img_height // self.transformer.patch_size
        patch_w = self.transformer.img_width // self.transformer.patch_size
        img_tokens_per_frame = patch_h * patch_w * len(self.config.obs_cam_keys)
        act_tokens_per_frame = self.config.image_frame_stride
        rdt_horizon = self.config.chunk_size
        rdt_x_pos_emb_config = [("act", rdt_horizon + self.config.rdt.num_register_tokens)]
        rdt_img_pos_emb_config = [("image", (self.config.window_size * len(self.config.obs_cam_keys), patch_h, patch_w))]
        rdt_act_pos_emb_config = [("action", (self.config.window_size, act_tokens_per_frame))]

        self.action_head = RDT(
            horizon=rdt_horizon,
            output_size=self.transformer.action_dim,
            config=rdt_config,
            x_pos_emb_config=rdt_x_pos_emb_config,
            lang_pos_emb_config=None,
            max_lang_len=0,
            img_pos_emb_config=rdt_img_pos_emb_config,
            max_img_len=self.config.window_size * img_tokens_per_frame,
            act_pos_emb_config=rdt_act_pos_emb_config,
            max_act_len=self.config.window_size * act_tokens_per_frame,
            dtype=self.dtype,
        )
        self.action_head.to(self.device)

        if config.gradient_checkpointing:
            self.transformer.gradient_checkpointing_enable()
        if config.long_context:
            self.transformer.fixed_input_length = False

        def _load_checkpoint_state(path):
            if str(path).endswith('.safetensors'):
                return load_file(path, device=str(self.device))
            state = torch.load(path, map_location=self.device)
            if isinstance(state, dict) and 'state_dict' in state and isinstance(state['state_dict'], dict):
                return state['state_dict']
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

        # ActionVGGT loading: prefer resume_from, fallback to pretrained.
        transformer_resume_from = getattr(config, 'transformer_resume_from', None)
        transformer_pretrained = getattr(config, 'transformer_pretrained', None)
        if transformer_resume_from:
            if config.rank == 0:
                logger.info(f"Resuming ActionVGGT from: {transformer_resume_from}")
            transformer_state = _load_checkpoint_state(transformer_resume_from)
            transformer_state = _adapt_transformer_state_for_resolution(self.transformer, transformer_state)
            logger.info(self.transformer.load_state_dict(transformer_state, strict=False))
        elif transformer_pretrained:
            logger.info(f"Loading ActionVGGT pretrained from: {transformer_pretrained}")
            transformer_state = _load_checkpoint_state(transformer_pretrained)
            transformer_state = _adapt_transformer_state_for_resolution(self.transformer, transformer_state)
            logger.info(self.transformer.load_state_dict(transformer_state, strict=False))

        # RDT action head loading: prefer resume_from, fallback to pretrained.
        action_head_resume_from = getattr(config, 'action_head_resume_from', None)
        action_head_pretrained = getattr(config, 'action_head_pretrained', None)
        if action_head_resume_from:
            if config.rank == 0:
                logger.info(f"Resuming RDT action head from: {action_head_resume_from}")
            action_head_state = _load_checkpoint_state(action_head_resume_from)
            logger.info(self.action_head.load_state_dict(action_head_state, strict=False))
        elif action_head_pretrained:
            logger.info(f"Loading RDT action head pretrained from: {action_head_pretrained}")
            logger.info(self.action_head.load_state_dict(_load_checkpoint_state(action_head_pretrained), strict=False))

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

        # freeze
        logger.info("Freezing patch embedding and positional encoding parameters...")
        frozen_params = 0
        total_params = 0

        frozen_param_names = []

        for name, param in self.transformer.named_parameters():
            total_params += param.numel()
            param.requires_grad = True

        if hasattr(self.transformer, 'aggregator') and hasattr(self.transformer.aggregator, 'patch_embed'):
            for param in self.transformer.aggregator.patch_embed.parameters():
                if param.requires_grad:
                    param.requires_grad = False

        if hasattr(self.transformer, 'aggregator') and hasattr(self.transformer.aggregator, 'camera_token'):
            self.transformer.aggregator.camera_token.requires_grad = False

        if hasattr(self.transformer, 'aggregator') and hasattr(self.transformer.aggregator, 'register_token'):
            self.transformer.aggregator.register_token.requires_grad = False


        for name, p in self.transformer.named_parameters():
            if not p.requires_grad:
                frozen_params += p.numel()
                frozen_param_names.append(name)

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

        self.save_dir = Path(config.save_root) / "ckpt"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        # if hasattr(config, 'resume_from') and config.resume_from:
        #     self._load_training_state(config.resume_from)
    
    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=B, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=1)
        targets =train_scheduler.training_target(latent, noise, timesteps)

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
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=1)
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
        """Prepare input dict following WAN-style structure but with raw images."""
        window_size = getattr(self.config, 'window_size', None)
        chunk_size = getattr(self.config, 'chunk_size', None)

        # Build image dict
        if 'images' not in batch_dict:
            raise ValueError("batch_dict must include raw 'images' for ActionVGGT training")

        images_full = batch_dict['images']  # [B, C_image, F, H, W]
        # cut image height and width to be divisible by patch size
        # patch_f, patch_h, patch_w = self.patch_size
        # H = (H // patch_h) * patch_h
        # W = (W // patch_w) * patch_w
        # images = images[:, :, :, :H, :W]
        B, _, F_total, H, W = images_full.shape
        # Build action dict
        actions_full = batch_dict["actions"] # [B, C_action, F, N, 1]
        B, _, F_total_action, N, _ = actions_full.shape

        if F_total != F_total_action:
            raise ValueError(f"images/actions frame count mismatch: {F_total} vs {F_total_action}")

                # Use action frame nums as chunk_size, instead of image frames
        min_t = max(window_size - 1, 0)
        max_t = max(batch_dict['num_frames'] - (chunk_size + 1) // N, min_t)
        if max_t < min_t:
            data_timestep = min_t if F_total > 0 else 0
        else:
            data_timestep = torch.randint(min_t, max_t + 1, (B,)).to(self.device)

        if not torch.is_tensor(data_timestep):
            data_timestep = torch.full((B,), int(data_timestep), device=self.device, dtype=torch.long)
        data_timestep = data_timestep.long()

        window_end_global = data_timestep + 1
        window_start_global = torch.clamp(window_end_global - window_size, min=0)

        if (window_start_global != window_start_global[0]).any() or (window_end_global != window_end_global[0]).any():
            raise ValueError("Variable window boundaries across batch are not supported")

        window_start_idx = int(window_start_global[0].item())
        window_end_idx = int(window_end_global[0].item())

        images = images_full[:, :, window_start_idx:window_end_idx]
        actions = actions_full[:, :, window_start_idx:window_end_idx]
        F = images.shape[2]
        local_data_timestep = data_timestep - window_start_global

        window_end = local_data_timestep + 1
        image_mask = torch.zeros_like(images, dtype=torch.bool)
        action_mask = torch.zeros_like(actions, dtype=torch.bool)
        image_mask[:, :, :window_end] = True
        action_mask[:, :, :window_end - 1] = True


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
            action_mask = action_mask,
        )

        # Build noised action chunks
        action_chunk_start_idx = data_timestep * N
        action_flat = rearrange(actions_full, 'b c f n 1 -> b c (f n)')
        max_action_tokens = action_flat.shape[-1]
        action_chunk_end_idx = torch.clamp(action_chunk_start_idx + chunk_size, max=max_action_tokens)

        if (action_chunk_start_idx != action_chunk_start_idx[0]).any() or (action_chunk_end_idx != action_chunk_end_idx[0]).any():
            raise ValueError("Variable action chunk boundaries across batch are not supported")

        action_chunk_start = int(action_chunk_start_idx[0].item())
        action_chunk_end = int(action_chunk_end_idx[0].item())
        action_chunk = action_flat[:, :, action_chunk_start:action_chunk_end]  # [B, C_action, chunk_size]
        action_chunk = rearrange(action_chunk, 'b c f -> b c f 1 1')  # [B, C_action, chunk_size, 1, 1]
        action_chunk_dict = self._add_noise(
            action_chunk,
            train_scheduler=self.train_scheduler_action,
            action_mask=None,
            action_mode=True,
        )
        action_chunk_dict["pred_frame_idx"] = local_data_timestep
        action_chunk_dict['targets'] = rearrange(action_chunk_dict['targets'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]
        action_chunk_dict['noisy_latents'] = rearrange(action_chunk_dict['noisy_latents'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]
        action_chunk_dict['latent'] = rearrange(action_chunk_dict['latent'], 'b c f 1 1 -> b c f')  # [B, C_action, chunk_size]
        
        input_dict = {
            'image_dict': image_dict,
            'action_dict': action_dict,
            'pred_action_chunk_dict': action_chunk_dict,
            'chunk_size': chunk_size,
            'window_size': window_size,
        }
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
            state_c = input_dict['pred_action_chunk_dict']['latent'][:, :, 0:1]  # [B, C_latent, 1]
            state_c = state_c.permute(0, 2, 1)  # [B, 1, C_latent]

            action_pred = self.action_head(
                action_chunk,
                timesteps,
                img_c=rdt_conds['rdt_img_c'],
                act_c=rdt_conds['rdt_act_c'],
                state_c=state_c,
                embed_input=True,
                decode_output=True,
            )

            action_loss = self.compute_loss(input_dict, action_pred)
            loss = action_loss

            loss.backward()

            # Accumulate losses for logging
            # accumulated_latent_losses.append(latent_loss.detach())
            accumulated_action_losses.append(action_loss.detach())

            # Only update weights after accumulating gradients
            if should_sync:
                total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
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

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []

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
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
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
            # optim_state = get_optimizer_state_dict(
            #         self.transformer, self.optimizer,
            #         options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            #     )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # Save action head in the same format as pretrained model
                action_head_dir = checkpoint_dir / "action_head"
                action_head_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving action head to {action_head_dir}")

                action_head_file = action_head_dir / "diffusion_pytorch_model.safetensors"
                save_file(action_head_state_dict_bf16, action_head_file)

                action_head_config_file = action_head_dir / "config.json"
                action_head_config_dict = dict(self.action_head.config)
                action_head_config_dict.pop('_name_or_path', None)
                with open(action_head_config_file, 'w') as f:
                    json.dump(action_head_config_dict, f, indent=2)

                # # Save optimizer state and training metadata in PyTorch format
                # training_state_path = checkpoint_dir / "training_state.pt"
                # logger.info(f"Saving training state to {training_state_path}")
                # torch.save({
                #     'step': self.step,
                #     'optimizer_state_dict': optim_state,
                #     'config': vars(self.config),
                # }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            if dist.is_initialized():
                dist.barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
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
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for FSDP)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
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

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

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

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()