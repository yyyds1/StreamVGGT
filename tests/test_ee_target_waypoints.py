import numpy as np

from evaluation.ee_target_waypoints import build_linear_ee_target_transitions


def test_fixed_distance_waypoints_match_training_slot_rule():
    current = np.zeros((2, 8), dtype=np.float32)
    target = np.zeros((2, 8), dtype=np.float32)
    target[0, :3] = [0.3, 0.0, 0.0]
    target[0, 7] = -1.0
    target[1, :3] = [0.0, 0.0, 0.3]
    target[1, 7] = 1.0

    transitions = build_linear_ee_target_transitions(current, target, 6)

    assert transitions.shape == (2, 6, 4)
    np.testing.assert_allclose(
        transitions[0, :, 0],
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        transitions[1, :, 2],
        [0.0, 0.05, 0.10, 0.15, 0.20, 0.25],
        atol=1e-6,
    )


def test_fixed_distance_waypoints_append_short_final_target_and_pad():
    current = np.zeros((2, 8), dtype=np.float32)
    target = np.zeros((2, 8), dtype=np.float32)
    target[0, 0] = 0.12

    transitions, valid, path_distances = build_linear_ee_target_transitions(
        current,
        target,
        6,
        waypoint_distance=0.05,
        return_metadata=True,
    )

    np.testing.assert_allclose(
        transitions[0, :, 0],
        [0.0, 0.05, 0.10, 0.12, 0.0, 0.0],
        atol=1e-6,
    )
    np.testing.assert_array_equal(valid[0], [True, True, True, True, False, False])
    np.testing.assert_allclose(path_distances[0], [0.0, 0.05, 0.10, 0.12, -1.0, -1.0])


def test_invalid_arm_holds_current_state():
    current = np.ones((2, 8), dtype=np.float32)
    target = np.full((2, 8), 3.0, dtype=np.float32)
    transitions = build_linear_ee_target_transitions(current, target, 4, valid=[True, False])

    np.testing.assert_allclose(transitions[0, 0], 0.0)
    np.testing.assert_allclose(transitions[1], 0.0)


def test_zero_position_distance_keeps_gripper_transition():
    current = np.zeros((2, 8), dtype=np.float32)
    target = np.zeros((2, 8), dtype=np.float32)
    target[0, 7] = -1.0

    transitions, valid, path_distances = build_linear_ee_target_transitions(
        current,
        target,
        6,
        return_metadata=True,
    )

    np.testing.assert_allclose(transitions[0, 0], [0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(transitions[0, 1:], 0.0)
    np.testing.assert_array_equal(valid[0], [True, False, False, False, False, False])
    np.testing.assert_allclose(path_distances[0], [0.0, -1.0, -1.0, -1.0, -1.0, -1.0])
