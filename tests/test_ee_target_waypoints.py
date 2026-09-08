import numpy as np

from evaluation.ee_target_waypoints import build_linear_ee_target_transitions


def test_linear_waypoints_reach_target_monotonically():
    current = np.zeros((2, 8), dtype=np.float32)
    target = np.zeros((2, 8), dtype=np.float32)
    target[0, :3] = [0.3, -0.2, 0.1]
    target[0, 7] = -1.0
    target[1, :3] = [-0.2, 0.1, 0.4]
    target[1, 7] = 1.0

    transitions = build_linear_ee_target_transitions(current, target, 6)

    assert transitions.shape == (2, 6, 4)
    np.testing.assert_allclose(transitions[:, -1], target[:, [0, 1, 2, 7]])
    assert np.all(np.diff(transitions[0, :, 0]) > 0)
    assert np.all(np.diff(transitions[1, :, 2]) > 0)


def test_invalid_arm_holds_current_state():
    current = np.ones((2, 8), dtype=np.float32)
    target = np.full((2, 8), 3.0, dtype=np.float32)
    transitions = build_linear_ee_target_transitions(current, target, 4, valid=[True, False])

    np.testing.assert_allclose(transitions[0, -1], [2.0, 2.0, 2.0, 2.0])
    np.testing.assert_allclose(transitions[1], 0.0)

