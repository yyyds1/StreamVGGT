import numpy as np
import pytest

from evaluation.robotwin.joint_state import compact_dual_arm_joint_state


def test_compact_dual_arm_joint_state_keeps_six_joints_and_gripper_per_arm():
    left = [0, 1, 2, 3, 4, 5, 0.25]
    right = [6, 7, 8, 9, 10, 11, 0.75]

    state = compact_dual_arm_joint_state(left, right)

    np.testing.assert_allclose(state, left + right)
    assert state.shape == (14,)
    assert state.dtype == np.float32


def test_compact_dual_arm_joint_state_drops_legacy_extra_joint():
    left = np.arange(8, dtype=np.float32)
    right = np.arange(10, 18, dtype=np.float32)

    state = compact_dual_arm_joint_state(left, right)

    np.testing.assert_allclose(state, [0, 1, 2, 3, 4, 5, 7, 10, 11, 12, 13, 14, 15, 17])


@pytest.mark.parametrize(
    "left,right,match",
    [
        ([0] * 6, [0] * 7, "left arm live joint state dimension 6"),
        ([0] * 7, [0] * 9, "right arm live joint state dimension 9"),
        ([0, 0, 0, 0, 0, np.nan, 0], [0] * 7, "non-finite"),
    ],
)
def test_compact_dual_arm_joint_state_rejects_invalid_live_state(left, right, match):
    with pytest.raises(ValueError, match=match):
        compact_dual_arm_joint_state(left, right)
