import torch
import torch.nn as nn
import torch.nn.functional as F

from streamvggt.layers import Mlp
from streamvggt.layers.block import Block


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


class EETargetHead(nn.Module):
    """CameraHead-style iterative decoder for dual-arm target EE pose."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        trunk_depth: int = 4,
        num_iterations: int = 4,
        init_values: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.target_dim = 8
        self.trunk_depth = int(trunk_depth)
        self.num_iterations = int(num_iterations)

        self.trunk = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    init_values=init_values,
                )
                for _ in range(self.trunk_depth)
            ]
        )
        self.token_norm = nn.LayerNorm(embed_dim)
        self.trunk_norm = nn.LayerNorm(embed_dim)
        self.empty_pose_tokens = nn.Parameter(torch.zeros(1, 2, self.target_dim))
        self.embed_pose = nn.Linear(self.target_dim, embed_dim)
        self.poseLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 3 * embed_dim, bias=True))
        self.adaln_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.pose_branch = Mlp(
            in_features=embed_dim,
            hidden_features=embed_dim // 2,
            out_features=self.target_dim,
            drop=0,
        )

    def _activate_target(self, pred_target: torch.Tensor) -> torch.Tensor:
        quat = F.normalize(pred_target[..., 3:7], dim=-1, eps=1e-6)
        return torch.cat([pred_target[..., :3], quat, pred_target[..., 7:8]], dim=-1)

    def _trunk_fn(self, pose_tokens: torch.Tensor, num_iterations: int) -> list[torch.Tensor]:
        batch_size, num_arms, _ = pose_tokens.shape
        pred_target = None
        pred_target_list = []

        for _ in range(num_iterations):
            if pred_target is None:
                module_input = self.embed_pose(self.empty_pose_tokens.expand(batch_size, num_arms, -1))
            else:
                module_input = self.embed_pose(pred_target.detach())

            shift_msa, scale_msa, gate_msa = self.poseLN_modulation(module_input).chunk(3, dim=-1)
            pose_tokens_modulated = gate_msa * modulate(self.adaln_norm(pose_tokens), shift_msa, scale_msa)
            pose_tokens_modulated = pose_tokens_modulated + pose_tokens

            for block in self.trunk:
                pose_tokens_modulated = block(pose_tokens_modulated)

            pred_delta = self.pose_branch(self.trunk_norm(pose_tokens_modulated))
            if pred_target is None:
                pred_target = pred_delta
            else:
                pred_target = pred_target + pred_delta

            pred_target_list.append(self._activate_target(pred_target))

        return pred_target_list

    def forward(self, ee_tokens: torch.Tensor, num_iterations: int | None = None) -> dict:
        if ee_tokens.ndim != 3:
            raise ValueError(f"ee_tokens must be [B, 2, D], got {tuple(ee_tokens.shape)}")
        if ee_tokens.shape[1] != 2:
            raise ValueError(f"ee_tokens must contain left/right tokens with shape [B, 2, D], got {tuple(ee_tokens.shape)}")

        if num_iterations is None:
            num_iterations = self.num_iterations

        pose_tokens = self.token_norm(ee_tokens)
        pred_target_list = self._trunk_fn(pose_tokens, num_iterations=num_iterations)
        pred = pred_target_list[-1]
        return {
            "ee_target": pred,
            "ee_target_list": pred_target_list,
            "left_ee_target": pred[:, 0],
            "right_ee_target": pred[:, 1],
        }
