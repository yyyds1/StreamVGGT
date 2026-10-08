"""Shared validity-mask construction for EE-target RDT condition tokens."""

from __future__ import annotations

import torch


def build_ee_target_token_mask(
    valid,
    sequence_len: int,
    *,
    batch_size: int,
    device=None,
) -> torch.Tensor:
    """Expand per-arm waypoint validity into the four-token RDT layout.

    The token order is waypoint-major:
    ``[left_xyz, left_gripper, right_xyz, right_gripper]`` per waypoint.
    The returned mask has shape ``[B, 4*K]`` and uses ``True`` for valid
    condition tokens.
    """
    sequence_len = max(1, int(sequence_len))
    batch_size = max(1, int(batch_size))

    if valid is None:
        waypoint_valid = torch.ones(
            (batch_size, 2, sequence_len),
            device=device,
            dtype=torch.bool,
        )
    else:
        waypoint_valid = torch.as_tensor(valid, device=device, dtype=torch.bool)
        if waypoint_valid.ndim == 1:
            if waypoint_valid.numel() != 2:
                raise ValueError(
                    "EE-target validity must contain two arm values when rank 1, "
                    f"got shape {tuple(waypoint_valid.shape)}"
                )
            waypoint_valid = waypoint_valid.reshape(1, 2, 1)
        elif waypoint_valid.ndim == 2:
            if waypoint_valid.shape[1] == 2 and waypoint_valid.shape[0] in {1, batch_size}:
                waypoint_valid = waypoint_valid.unsqueeze(-1)
            elif waypoint_valid.shape[0] == 2:
                waypoint_valid = waypoint_valid.unsqueeze(0)
            else:
                raise ValueError(
                    "EE-target validity must be [B,2], [B,2,K], or [2,K], "
                    f"got shape {tuple(waypoint_valid.shape)}"
                )
        elif waypoint_valid.ndim == 3:
            pass
        elif waypoint_valid.ndim == 4 and waypoint_valid.shape[-1] == 1:
            waypoint_valid = waypoint_valid[..., 0]
        else:
            raise ValueError(
                "EE-target validity must be [B,2], [B,2,K], [2,K], or [B,2,K,1], "
                f"got shape {tuple(waypoint_valid.shape)}"
            )

        if waypoint_valid.ndim != 3 or waypoint_valid.shape[1] != 2:
            raise ValueError(
                "Normalized EE-target validity must have shape [B,2,K], "
                f"got {tuple(waypoint_valid.shape)}"
            )
        if waypoint_valid.shape[0] == 1 and batch_size != 1:
            waypoint_valid = waypoint_valid.expand(batch_size, -1, -1)
        elif waypoint_valid.shape[0] != batch_size:
            raise ValueError(
                "EE-target validity batch mismatch: "
                f"got {waypoint_valid.shape[0]}, expected {batch_size}"
            )
        if waypoint_valid.shape[2] == 1 and sequence_len != 1:
            waypoint_valid = waypoint_valid.expand(-1, -1, sequence_len)
        elif waypoint_valid.shape[2] != sequence_len:
            raise ValueError(
                "EE-target validity sequence mismatch: "
                f"got {waypoint_valid.shape[2]}, expected {sequence_len}"
            )

    # [B, 2, K] -> [B, K, 2, 2] -> [B, 4*K].
    return (
        waypoint_valid.permute(0, 2, 1)
        .unsqueeze(-1)
        .expand(-1, -1, -1, 2)
        .reshape(batch_size, 4 * sequence_len)
    )
