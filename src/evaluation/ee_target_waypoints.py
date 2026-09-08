"""Evaluation-time EE waypoint construction utilities."""

from __future__ import annotations

import numpy as np


def build_linear_ee_target_transitions(
    current_ee,
    expert_target,
    sequence_len,
    *,
    valid=None,
):
    """Interpolate K EE transition targets from the current state to the expert target.

    Both inputs use ``[left, right, xyz(3), quaternion(4), gripper]`` per arm.
    The returned condition deliberately contains only ``[dx, dy, dz, dgripper]``;
    orientation remains the responsibility of the action head.

    Invalid expert targets become zero transitions, which is equivalent to holding
    the corresponding arm at its current state.
    """
    current = np.asarray(current_ee, dtype=np.float32)
    target = np.asarray(expert_target, dtype=np.float32)
    if current.shape != (2, 8):
        raise ValueError(f"current_ee must have shape [2, 8], got {current.shape}")
    if target.shape != (2, 8):
        raise ValueError(f"expert_target must have shape [2, 8], got {target.shape}")

    horizon = max(1, int(sequence_len))
    valid_mask = np.ones((2,), dtype=bool) if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if valid_mask.shape != (2,):
        raise ValueError(f"valid must have shape [2], got {valid_mask.shape}")

    delta = np.stack(
        [target[:, :3] - current[:, :3], target[:, 7] - current[:, 7]],
        axis=-1,
    )
    delta[~valid_mask] = 0.0
    alpha = np.arange(1, horizon + 1, dtype=np.float32) / float(horizon)
    return delta[:, None, :] * alpha[None, :, None]

