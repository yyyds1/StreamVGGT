import logging
from typing import Optional

import torch
import torch.nn as nn

from streamvggt.layers import PatchEmbed
from streamvggt.layers.block import Block
from streamvggt.layers.vision_transformer import vit_base, vit_giant2, vit_large, vit_small

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class VGAAggregator(nn.Module):
    """VGA aggregator with modality-specific frame attention and shared global attention.

    Frame attention policy:
    1) image tokens: frame-attn per camera view
    2) action tokens: frame-attn inside action branch only
    3) language tokens: self-only (kept unchanged in frame step)
    4) merged tokens: global-attn on concatenated sequence
    """

    def __init__(
        self,
        img_height=224,
        img_width=224,
        patch_size=14,
        embed_dim=1024,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        qk_norm=True,
        init_values=0.01,
        action_dim=30,
        action_chunk_size=24,
        num_image_views=1,
        image_frame_stride=4,
    ):
        super().__init__()
        self.image_height = img_height
        self.image_width = img_width
        self.num_image_views = max(1, int(num_image_views))
        self.image_frame_stride = max(1, int(image_frame_stride))
        self.patch_size = patch_size
        self.depth = int(depth)
        self.action_chunk_size = int(action_chunk_size)

        self.__build_patch_embed__(
            patch_embed,
            img_height,
            img_width,
            patch_size,
            num_register_tokens,
            embed_dim=embed_dim,
        )

        # Learnable action query tokens [C, D], where C is action chunk size.
        self.action_query_tokens = nn.Parameter(torch.randn(self.action_chunk_size, embed_dim))
        nn.init.normal_(self.action_query_tokens, std=1e-6)

        self.frame_blocks_image = nn.ModuleList(
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
                    rope=None,
                )
                for _ in range(self.depth)
            ]
        )
        self.frame_blocks_action = nn.ModuleList(
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
                    rope=None,
                )
                for _ in range(self.depth)
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
                    rope=None,
                )
                for _ in range(self.depth)
            ]
        )

        # Two slots: first-frame token and remaining-frame token.
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        self.token_idx = dict(camera=(0, 1), register=(1, 1 + num_register_tokens))

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
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
        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=(img_height, img_width),
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
            return

        vit_models = {
            "dinov2_vitl14_reg": vit_large,
            "dinov2_vitb14_reg": vit_base,
            "dinov2_vits14_reg": vit_small,
            "dinov2_vitg2_reg": vit_giant2,
        }
        self.patch_embed = vit_models[patch_embed](
            img_size=(img_height, img_width),
            patch_size=patch_size,
            num_register_tokens=num_register_tokens,
            interpolate_antialias=interpolate_antialias,
            interpolate_offset=interpolate_offset,
            block_chunks=block_chunks,
            init_values=init_values,
        )
        if hasattr(self.patch_embed, "mask_token"):
            self.patch_embed.mask_token.requires_grad_(False)

    def _split_views(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, S, 3, H, W] -> [B, S, V, 3, H_v, W_v]
        bsz, seq_len, c_in, h, w = images.shape
        if self.num_image_views == 1:
            return images.unsqueeze(2)

        if h % self.num_image_views == 0 and (h // self.num_image_views) == self.image_height:
            return images.reshape(bsz, seq_len, c_in, self.num_image_views, self.image_height, w).permute(0, 1, 3, 2, 4, 5)
        if w % self.num_image_views == 0 and (w // self.num_image_views) == self.image_width:
            return images.reshape(bsz, seq_len, c_in, h, self.num_image_views, self.image_width).permute(0, 1, 4, 2, 3, 5)

        raise ValueError(
            "Unable to split multi-view image tensor. "
            f"Got shape {(bsz, seq_len, c_in, h, w)} and num_image_views={self.num_image_views}."
        )

    def forward(
        self,
        images: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        text_emb: Optional[torch.Tensor] = None,
        image_grid_id=None,
        action_grid_id=None,
        return_all_layers: bool = True,
        **kwargs,
    ):
        del image_grid_id, action_grid_id, kwargs

        bsz, seq_len, c_in, _, _ = images.shape
        if c_in != 3:
            raise ValueError(f"Expected 3-channel images, got {c_in}")

        images = (images - self._resnet_mean.to(device=images.device, dtype=images.dtype)) / self._resnet_std.to(
            device=images.device, dtype=images.dtype
        )
        view_images = self._split_views(images)
        _, _, num_views, _, h_view, w_view = view_images.shape

        flat_view_images = view_images.reshape(bsz * seq_len * num_views, 3, h_view, w_view)
        patch_tokens = self.patch_embed(flat_view_images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]
        patch_tokens = patch_tokens.reshape(bsz, seq_len, num_views, patch_tokens.shape[1], patch_tokens.shape[2])
        patches_per_view = patch_tokens.shape[3]

        del actions
        action_tokens = self.action_query_tokens[None, None].expand(bsz, seq_len, -1, -1)
        n_action_tokens = action_tokens.shape[2]

        if text_emb is None:
            lang_token = torch.zeros(bsz, 1, action_tokens.shape[-1], device=images.device, dtype=images.dtype)
        else:
            if text_emb.ndim == 2:
                pooled = text_emb
            elif text_emb.ndim == 3:
                pooled = text_emb.mean(dim=1)
            else:
                raise ValueError(f"text_emb must be [B,D] or [B,L,D], got shape {tuple(text_emb.shape)}")
            lang_token = pooled.unsqueeze(1).to(images.dtype)
        lang_tokens = lang_token.unsqueeze(1).repeat(1, seq_len, 1, 1)

        camera_token = slice_expand_and_flatten(self.camera_token, bsz, seq_len).reshape(bsz, seq_len, 1, -1)
        register_token = slice_expand_and_flatten(self.register_token, bsz, seq_len).reshape(
            bsz, seq_len, self.token_idx["register"][1] - self.token_idx["register"][0], -1
        )

        image_branch_special = torch.cat([camera_token, register_token], dim=2)
        image_branch_tokens = patch_tokens.reshape(bsz, seq_len, num_views * patches_per_view, patch_tokens.shape[-1])

        # Build frame-level modality branches.
        outputs = []
        last_output = None
        for layer_idx in range(self.depth):
            updated_view_tokens = []
            updated_special_tokens = []
            for view_idx in range(num_views):
                view_patches = patch_tokens[:, :, view_idx]  # [B, S, P_v, C]
                view_input = torch.cat([image_branch_special, view_patches], dim=2).reshape(
                    bsz * seq_len, image_branch_special.shape[2] + patches_per_view, -1
                )
                view_output = self.frame_blocks_image[layer_idx](view_input)
                view_output = view_output.reshape(bsz, seq_len, image_branch_special.shape[2] + patches_per_view, -1)
                updated_special_tokens.append(view_output[:, :, : image_branch_special.shape[2]])
                updated_view_tokens.append(view_output[:, :, image_branch_special.shape[2] :])

            image_branch_special = torch.stack(updated_special_tokens, dim=0).mean(dim=0)
            patch_tokens = torch.stack(updated_view_tokens, dim=2)
            image_branch_tokens = patch_tokens.reshape(bsz, seq_len, num_views * patches_per_view, patch_tokens.shape[-1])

            action_branch = action_tokens.permute(0, 2, 1, 3).reshape(bsz * n_action_tokens, seq_len, action_tokens.shape[-1])
            action_branch = self.frame_blocks_action[layer_idx](action_branch)
            action_tokens = action_branch.reshape(bsz, n_action_tokens, seq_len, -1).permute(0, 2, 1, 3)

            # language frame-attn self-only: keep tokens unchanged.
            merged = torch.cat([image_branch_special, image_branch_tokens, action_tokens, lang_tokens], dim=2)

            p_total = merged.shape[2]
            global_in = merged.reshape(bsz, seq_len * p_total, -1)
            global_out = self.global_blocks[layer_idx](global_in)
            global_out = global_out.reshape(bsz, seq_len, p_total, -1)

            # Keep StreamVGGT-compatible geometry feature format: [frame || global].
            layer_out = torch.cat([merged, global_out], dim=-1)
            if return_all_layers:
                outputs.append(layer_out)
            else:
                last_output = layer_out

        if not return_all_layers:
            outputs = [last_output]

        n_special = image_branch_special.shape[2]
        self.token_idx["camera"] = (0, 1)
        self.token_idx["register"] = (1, n_special)
        self.token_idx["image"] = (n_special, n_special + num_views * patches_per_view)
        self.token_idx["action"] = (
            self.token_idx["image"][1],
            self.token_idx["image"][1] + n_action_tokens,
        )
        self.token_idx["lang"] = (self.token_idx["action"][1], self.token_idx["action"][1] + 1)
        return outputs, self.token_idx


def slice_expand_and_flatten(token_tensor, bsz, seq_len):
    query = token_tensor[:, :1, ...].expand(bsz, 1, *token_tensor.shape[2:])
    others = token_tensor[:, 1:seq_len, ...].expand(bsz, max(seq_len - 1, 0), *token_tensor.shape[2:])
    combined = torch.cat([query, others], dim=1)
    return combined.reshape(bsz * seq_len, *combined.shape[2:])
