import numpy as np


def compact_dual_arm_joint_state(left_state, right_state):
    """Return RobotWin's dual-arm policy state as 6 joints + gripper per arm."""

    def compact_arm(value, arm_name):
        arm_state = np.asarray(value, dtype=np.float32).reshape(-1)
        if arm_state.size == 7:
            compact = arm_state
        elif arm_state.size == 8:
            compact = np.concatenate([arm_state[:6], arm_state[7:8]])
        else:
            raise ValueError(
                f"Unexpected {arm_name} live joint state dimension {arm_state.size}. "
                "Expected 7 values (6 arm joints + gripper) or 8 values "
                "(7 arm joints + gripper)."
            )
        if not np.all(np.isfinite(compact)):
            raise ValueError(f"{arm_name} live joint state contains non-finite values: {compact!r}")
        return compact.astype(np.float32, copy=False)

    left = compact_arm(left_state, "left arm")
    right = compact_arm(right_state, "right arm")
    state = np.concatenate([left, right]).astype(np.float32, copy=False)
    if state.size != 14:
        raise ValueError(f"RobotWin joint observation must be 14D, got {state.size}D")
    return state
