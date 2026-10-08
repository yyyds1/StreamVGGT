"""Evaluation-time EE waypoint construction utilities."""

from __future__ import annotations

import numpy as np


def build_linear_ee_target_transitions(
    current_ee,
    expert_target,
    sequence_len,
    *,
    valid=None,
    waypoint_distance=0.05,
    return_metadata=False,
):
    """Build fixed-distance EE transition targets from the current state.

    Both inputs use ``[left, right, xyz(3), quaternion(4), gripper]`` per arm.
    The returned condition deliberately contains only ``[dx, dy, dz, dgripper]``;
    orientation remains the responsibility of the action head.

    Evaluation does not have the recorded expert endpose trace available, so the
    current state-to-target segment is treated as a straight path. The slot
    construction mirrors ``pkl2lerobot._distance_along_path_samples``:

    ``[0, waypoint_distance, 2 * waypoint_distance, ..., target]``

    with at most ``sequence_len`` slots. If the target does not fit in the
    remaining slot, it is omitted and the sequence is padded with invalid slots.
    This intentionally preserves the training exporter semantics for long
    segments.

    If ``return_metadata`` is true, return ``(transitions, slot_valid,
    path_distances)``. ``path_distances`` is ``-1`` for padded slots.
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
    waypoint_distance = float(waypoint_distance)
    if not np.isfinite(waypoint_distance) or waypoint_distance < 0.0:
        raise ValueError(
            f"waypoint_distance must be a finite non-negative scalar, got {waypoint_distance}"
        )

    delta = np.concatenate(
        [target[:, :3] - current[:, :3], target[:, 7:8] - current[:, 7:8]],
        axis=-1,
    )
    transitions = np.zeros((2, horizon, 4), dtype=np.float32)
    slot_valid = np.zeros((2, horizon), dtype=bool)
    path_distances = np.full((2, horizon), -1.0, dtype=np.float32)

    for arm_idx in range(2):
        if not valid_mask[arm_idx]:
            continue

        total_distance = float(np.linalg.norm(delta[arm_idx, :3]))
        if total_distance <= 1e-9:
            # This is also the exporter behavior for a degenerate segment:
            # populate one slot with the expert target.
            transitions[arm_idx, 0] = delta[arm_idx]
            slot_valid[arm_idx, 0] = True
            path_distances[arm_idx, 0] = 0.0
            continue
        if waypoint_distance <= 1e-9:
            # The exporter treats a disabled distance sampler as one target
            # sample, regardless of the target's displacement.
            distances = [total_distance]
        else:
            distances = [0.0]
            next_distance = waypoint_distance
            while next_distance < total_distance - 1e-9 and len(distances) < horizon:
                distances.append(next_distance)
                next_distance += waypoint_distance
            if len(distances) < horizon:
                distances.append(total_distance)

        for slot_idx, distance in enumerate(distances[:horizon]):
            alpha = 0.0 if total_distance <= 1e-9 else distance / total_distance
            transitions[arm_idx, slot_idx] = delta[arm_idx] * np.float32(alpha)
            slot_valid[arm_idx, slot_idx] = True
            path_distances[arm_idx, slot_idx] = np.float32(distance)

    if return_metadata:
        return transitions, slot_valid, path_distances
    return transitions
