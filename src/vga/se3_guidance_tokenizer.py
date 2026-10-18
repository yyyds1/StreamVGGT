"""Encode sparse waypoint blocks as structured SE(3) guidance tokens."""
from __future__ import annotations

import torch
import torch.nn as nn


class SE3GuidanceTokenizer(nn.Module):
    """Encode each waypoint as left/right pose, gripper, and duration tokens."""

    TOKENS_PER_BLOCK = 7
    MAX_WAYPOINTS = 6
    FOURIER_BANDS = 4
    QUAT_EPS = 1e-8

    def __init__(self, embed_dim: int, max_duration: int = 32):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_duration = max(1.0, float(max_duration))
        xyz_dim = 3 + 2 * self.FOURIER_BANDS * 3
        self.left_position_proj = nn.Linear(xyz_dim, embed_dim)
        self.left_rotation_proj = nn.Linear(4, embed_dim)
        self.left_gripper_proj = nn.Linear(1, embed_dim)
        self.right_position_proj = nn.Linear(xyz_dim, embed_dim)
        self.right_rotation_proj = nn.Linear(4, embed_dim)
        self.right_gripper_proj = nn.Linear(1, embed_dim)
        self.duration_proj = nn.Linear(1, embed_dim)

    @classmethod
    def _fourier_encode_xyz(cls, xyz: torch.Tensor) -> torch.Tensor:
        freq = (2.0 ** torch.arange(
            cls.FOURIER_BANDS, device=xyz.device, dtype=xyz.dtype
        )) * torch.pi
        angles = xyz[..., None] * freq
        sincos = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return torch.cat([xyz, sincos.flatten(start_dim=-2)], dim=-1)

    @classmethod
    def _normalize_quaternion(cls, quaternion: torch.Tensor) -> torch.Tensor:
        return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(cls.QUAT_EPS)

    @classmethod
    def _relative_quaternion(cls, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return q_current^-1 * q_target, with a canonical hemisphere."""
        current = cls._normalize_quaternion(current)
        target = cls._normalize_quaternion(target)
        target = torch.where((current * target).sum(dim=-1, keepdim=True) < 0, -target, target)
        cw, cx, cy, cz = current.unbind(dim=-1)
        tw, tx, ty, tz = target.unbind(dim=-1)
        delta = torch.stack((
            cw * tw + cx * tx + cy * ty + cz * tz,
            cw * tx - cx * tw - cy * tz + cz * ty,
            cw * ty + cx * tz - cy * tw - cz * tx,
            cw * tz - cx * ty + cy * tx - cz * tw,
        ), dim=-1)
        delta = cls._normalize_quaternion(delta)
        return torch.where(delta[..., :1] < 0, -delta, delta)

    def forward(self, guidance: dict, dtype=None):
        required = (
            "left_pose_7d", "right_pose_7d", "left_gripper", "right_gripper",
            "duration", "current_ee",
        )
        missing = [key for key in required if key not in guidance]
        if missing:
            raise KeyError(f"SE3 guidance is missing fields: {missing}")

        device = guidance["left_pose_7d"].device
        model_dtype = self.left_position_proj.weight.dtype
        left_pose = guidance["left_pose_7d"].to(device=device, dtype=model_dtype)
        right_pose = guidance["right_pose_7d"].to(device=device, dtype=model_dtype)
        left_gripper = guidance["left_gripper"].to(device=device, dtype=model_dtype)
        right_gripper = guidance["right_gripper"].to(device=device, dtype=model_dtype)
        duration = guidance["duration"].to(device=device, dtype=model_dtype)
        current_ee = guidance["current_ee"].to(device=device, dtype=model_dtype)

        if left_pose.ndim != 3 or left_pose.shape[-1] != 7:
            raise ValueError(f"left_pose_7d must be [B,K,7], got {tuple(left_pose.shape)}")
        if right_pose.shape != left_pose.shape:
            raise ValueError("left_pose_7d and right_pose_7d must have the same shape")
        if current_ee.ndim != 3 or current_ee.shape[1:] != (2, 8):
            raise ValueError(f"current_ee must be [B,2,8], got {tuple(current_ee.shape)}")

        batch, count = left_pose.shape[:2]
        if count > self.MAX_WAYPOINTS:
            raise ValueError(f"At most {self.MAX_WAYPOINTS} waypoint blocks are supported, got {count}")
        for name, value in (("left_gripper", left_gripper), ("right_gripper", right_gripper), ("duration", duration)):
            if value.shape != (batch, count):
                raise ValueError(f"{name} must have shape [B,K], got {tuple(value.shape)}")

        left_position_delta = left_pose[..., :3] - current_ee[:, None, 0, :3]
        right_position_delta = right_pose[..., :3] - current_ee[:, None, 1, :3]
        left_rotation_delta = self._relative_quaternion(current_ee[:, None, 0, 3:7], left_pose[..., 3:7])
        right_rotation_delta = self._relative_quaternion(current_ee[:, None, 1, 3:7], right_pose[..., 3:7])
        left_gripper_delta = left_gripper - current_ee[:, None, 0, 7]
        right_gripper_delta = right_gripper - current_ee[:, None, 1, 7]
        normalized_duration = (duration / self.max_duration).clamp(0.0, 1.0)

        tokens = torch.stack([
            self.left_position_proj(self._fourier_encode_xyz(left_position_delta)),
            self.left_rotation_proj(left_rotation_delta),
            self.left_gripper_proj(left_gripper_delta.unsqueeze(-1)),
            self.right_position_proj(self._fourier_encode_xyz(right_position_delta)),
            self.right_rotation_proj(right_rotation_delta),
            self.right_gripper_proj(right_gripper_delta.unsqueeze(-1)),
            self.duration_proj(normalized_duration.unsqueeze(-1)),
        ], dim=2)

        left_valid = guidance.get("left_valid", torch.ones((batch, count), device=device, dtype=torch.bool)).to(device=device, dtype=torch.bool)
        right_valid = guidance.get("right_valid", torch.ones((batch, count), device=device, dtype=torch.bool)).to(device=device, dtype=torch.bool)
        block_valid = guidance.get("block_valid", left_valid | right_valid).to(device=device, dtype=torch.bool)
        if left_valid.shape != (batch, count) or right_valid.shape != (batch, count) or block_valid.shape != (batch, count):
            raise ValueError("left_valid, right_valid, and block_valid must have shape [B,K]")
        token_mask = torch.stack([
            left_valid, left_valid, left_valid,
            right_valid, right_valid, right_valid,
            block_valid,
        ], dim=2).flatten(1, 2)
        return tokens.flatten(1, 2).to(dtype=dtype or model_dtype), token_mask
