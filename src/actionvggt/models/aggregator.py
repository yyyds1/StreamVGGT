# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, List, Dict, Any

from actionvggt.layers import PatchEmbed
from actionvggt.layers.block import Block, CrossBlock
from actionvggt.layers.rope import RotaryPositionEmbedding3D, PositionGetter
from actionvggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
from einops import rearrange

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.


    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
        action_dim (int): Dimensionality of action vectors for flow-matching.
        text_dim (int): Dimensionality of text embeddings for cross-attention.
    """

    def __init__(
        self,
        img_height=518,
        img_width=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        action_dim=30,
        text_dim=4096,
    ):
        super().__init__()
        self.image_height = img_height
        self.image_width = img_width
        self._frame_causal_mask_cache = {}
        self.__build_patch_embed__(patch_embed, img_height, img_width, patch_size, num_register_tokens, embed_dim=embed_dim)

        # Initialize rotary position embedding if frequency > 0
        self.rope = RotaryPositionEmbedding3D(frequency=rope_freq) if rope_freq > 0 else None
        # keep 2D position getter for fallback/debug only
        self.position_getter = PositionGetter() if self.rope is not None else None

        # Initialize action embedder if action_dim > 0
        if action_dim > 0:
            self.action_embedder = nn.Linear(action_dim, embed_dim)
            self.action_decoder = nn.Linear(embed_dim, action_dim)


        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        if "cross" in aa_order:
            self.cross_blocks = nn.ModuleList(
                [
                    CrossBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        proj_bias=proj_bias,
                        ffn_bias=ffn_bias,
                        init_values=init_values,
                        qk_norm=qk_norm,
                        rope=self.rope,
                    )
                    for _ in range(depth)
                ]
            )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})")

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        self.token_idx = dict(
            camera=(0, 1),
            register=(1, 1 + num_register_tokens),
        )

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.FloatTensor(value).reshape(1, 1, 3, 1, 1),
                persistent=False,
            )


    def __build_patch_embed__(
        self,
        patch_embed,
        img_height,
        img_width,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_height=img_height, img_width=img_width, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_height=img_height,
                img_width=img_width,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor,
        actions: Optional[torch.Tensor],
        text_emb: Optional[torch.Tensor] = None,
        # image_mask: Optional[torch.Tensor] = None,
        # action_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        use_cache=False,
        past_frame_idx=0,
        image_grid_id: Optional[torch.Tensor] = None,
        action_grid_id: Optional[torch.Tensor] = None,
        return_all_layers: bool = True,
     ) -> Tuple[List[torch.Tensor], int, Optional[Dict]]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, C, H, W], in range [0, 1].
                B: batch size, S: sequence length, C: RGB channels, H: height, W: width
           actions (Optional[torch.Tensor]): Action vectors with shape [B, C_act, S, N, 1].
               If provided, will be embedded and interleaved with image tokens.
           text_emb (Optional[torch.Tensor]): Text embeddings with shape [B, L_text, text_dim] for cross-attention.
           image_grid_id (Optional[torch.Tensor]): 3D grid id for image tokens, shape [B, 4, F*H*W].
           action_grid_id (Optional[torch.Tensor]): 3D grid id for action tokens, shape [B, 4, F*H*W] or [B, 4, F].

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        B, S, C_in, H, W = images.shape

        if use_cache and past_key_values[0] is not None:
            _, _, S_true, _, _ = past_key_values[0][0].shape
            S_true += 1
        else:
            S_true = S
        
        if use_cache and S > 1:
            print(f"Use KV cache expects S=1, got S={S}")

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}. Image shape: {images.shape}")

        # Normalize images while preserving dtype for mixed-precision runs.
        images = (images - self._resnet_mean.to(device=images.device, dtype=images.dtype)) / self._resnet_std.to(device=images.device, dtype=images.dtype)

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.reshape(-1, C_in, self.image_height, self.image_width)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape
        patch_tokens = patch_tokens.reshape(B * S, -1, C)

        # Process action tokens if provided
        action_vec = rearrange(actions, 'b c f h w -> (b f) (h w) c')
        action_tokens = self.action_embedder(action_vec)
        # print(f"actions: {actions.shape}, action_vec: {action_vec.shape}, action_tokens: {action_tokens.shape}")

        if use_cache:
            camera_token_full = slice_expand_and_flatten(self.camera_token, B, S_true)
            camera_token = camera_token_full[-1:, :, :]
            
            register_token_full = slice_expand_and_flatten(self.register_token, B, S_true)
            register_token = register_token_full[-1:, :, :]
        else:
            camera_token = slice_expand_and_flatten(self.camera_token, B, S)
            register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        # Interleave tokens temporally
        img_tokens = patch_tokens
        # Concatenate along token dimension for each frame
        # print(f"Camera token shape: {camera_token.shape}, Register token shape: {register_token.shape}, Image tokens shape: {img_tokens.shape}, Action tokens shape: {action_tokens.shape}")
        tokens = torch.cat([camera_token, register_token, img_tokens, action_tokens], dim=1)

        # Build 3D positions from grid_id (f/h/w). If not provided, fall back to 2D positions.
        pos = None
        if self.rope is not None:
            if image_grid_id is None:
                # fallback: create 2D positions and pad with zeros for time
                pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)
                pos = F.pad(pos, (0, 1), value=0)  # [B*S, H*W, 3]
            else:
                # image_grid_id expected [B, 4, F*H*W], use f/h/w only
                grid = image_grid_id[:, :3]  # [B, 3, F*H*W]
                grid = grid.reshape(B, 3, S, -1).permute(0, 2, 3, 1)  # [B, S, H*W, 3]
                pos = grid.reshape(B * S, -1, 3)

            # no special tokens; patch_start_idx is 0

            if action_tokens is not None:
                if action_grid_id is None:
                    action_pos = torch.zeros(B * S, 1, 3, device=images.device, dtype=pos.dtype)
                else:
                    # action_grid_id expected [B, 4, F*H*W] or [B, 4, F]
                    act_grid = action_grid_id[:, :3]
                    act_grid = act_grid.reshape(B, 3, S, -1).permute(0, 2, 3, 1)
                    action_pos = act_grid.reshape(B * S, -1, 3)
                pos = torch.cat([pos, action_pos], dim=1)

        self.token_idx["image"] = (self.token_idx["register"][1], self.token_idx["register"][1] + patch_tokens.shape[1])
        self.token_idx["action"] = (self.token_idx["image"][1], self.token_idx["image"][1] + action_tokens.shape[1])

        if self.token_idx["image"][0] > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.token_idx["image"][0], 3).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        _, P, C = tokens.shape


        frame_idx = 0
        global_idx = 0
        cross_idx = 0
        last_output = None

        for _ in range(self.aa_block_num):
            for attn_type in self.aa_order:
                # print(f"Processing block group with attention type: {attn_type}, frame_idx: {frame_idx}, global_idx: {global_idx}")
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    if use_cache:
                        if past_key_values[global_idx] is not None:
                            k, v = past_key_values[global_idx]
                        tokens, global_idx, global_intermediates, new_kv = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos,
                            past_key_values_block=past_key_values[global_idx] if past_key_values[global_idx] is not None else None,
                            use_cache=True,
                            past_frame_idx=past_frame_idx
                        )
                        past_key_values[global_idx - 1] = new_kv
                    else: 
                        tokens, global_idx, global_intermediates = self._process_global_attention(
                            tokens, B, S, P, C, global_idx, pos=pos
                        )
                elif attn_type == "cross":
                    tokens, cross_idx, cross_intermediates = self._process_cross_attention(
                        tokens, B, S, P, C, cross_idx, text_emb=text_emb
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")
            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                # concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i], cross_intermediates[i]], dim=-1)
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                if return_all_layers:
                    if last_output is None:
                        last_output = []
                    last_output.append(concat_inter)
                else:
                    last_output = concat_inter

        if return_all_layers:
            output = last_output if last_output is not None else []
        else:
            output = [last_output] if last_output is not None else []

        del concat_inter
        del frame_intermediates
        del global_intermediates
        if use_cache:
            return output, self.token_idx, past_key_values
        return output, self.token_idx

    def _get_frame_causal_mask(self, seq_len, tokens_per_frame, device):
        key = (seq_len, tokens_per_frame, device)
        mask = self._frame_causal_mask_cache.get(key)
        if mask is None:
            frame_ids = torch.arange(seq_len, device=device) // tokens_per_frame
            mask = frame_ids.unsqueeze(1) >= frame_ids.unsqueeze(0)
            self._frame_causal_mask_cache[key] = mask
        return mask

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
       
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 3):
            pos = pos.reshape(B, S, P, 3).reshape(B * S, P, 3)

        intermediates = []

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](
                tokens,
                pos=pos,
            )
            frame_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        past_key_values_block=None,
        use_cache=False,
        past_frame_idx=0
    ) -> Union[Tuple[torch.Tensor, int, List[torch.Tensor]], Tuple[torch.Tensor, int, List[torch.Tensor], List]]:
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
       
                """
        
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 3):
            pos = pos.reshape(B, S, P, 3).reshape(B, S * P, 3)
            
        intermediates = []

        for _ in range(self.aa_block_size):
            if not use_cache:
                L = S * P
                # Boolean masks keep the original frame-causal semantics while
                # avoiding the large additive mask tensor that pushes SDPA onto
                # a more memory-hungry path.
                attn_mask = self._get_frame_causal_mask(L, P, tokens.device)
            else:
                attn_mask = None
                
            if use_cache:
                tokens, block_kv = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                    past_key_values=past_key_values_block,
                    use_cache=True,
                )
            else:
                tokens = self.global_blocks[global_idx](
                    tokens,
                    pos=pos,
                    attn_mask=attn_mask,
                )
            global_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

            # if self.use_causal_global:
            #     del attn_mask
        if use_cache:
            return tokens, global_idx, intermediates, block_kv
        return tokens, global_idx, intermediates

    def _process_cross_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        cross_idx,
        text_emb=None,
    ) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
        """Separate cross-attention stage over text embeddings."""
        if tokens.shape != (B, S * P, C):
            tokens = tokens.reshape(B, S, P, C).reshape(B, S * P, C)

        intermediates = []

        for _ in range(self.aa_block_size):
            if text_emb is None:
                intermediates.append(tokens.reshape(B, S, P, C))
                cross_idx += 1
                continue

            tokens = self.cross_blocks[cross_idx](
                tokens,
                context=text_emb,
                pos=None,
                attn_mask=None,
            )
            cross_idx += 1
            intermediates.append(tokens.reshape(B, S, P, C))

        return tokens, cross_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    # Slice out the "query" tokens => shape (1, 1, ...)
    query = token_tensor[:, :1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:S, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.reshape(B * S, *combined.shape[2:])
    return combined
