import torch

from vga.ee_target_mask import build_ee_target_token_mask


def test_ee_target_mask_expands_waypoint_validity_in_token_order():
    valid = torch.tensor(
        [
            [
                [True, False, True],
                [False, True, False],
            ]
        ]
    )

    mask = build_ee_target_token_mask(
        valid,
        sequence_len=3,
        batch_size=1,
    )

    # Token order per waypoint is:
    # left_xyz, left_gripper, right_xyz, right_gripper.
    expected = torch.tensor(
        [[True, True, False, False, False, False, True, True, True, True, False, False]]
    )
    assert mask.shape == (1, 12)
    torch.testing.assert_close(mask, expected)


def test_ee_target_mask_supports_arm_level_validity_and_batch_broadcast():
    mask = build_ee_target_token_mask(
        torch.tensor([[True, False]]),
        sequence_len=2,
        batch_size=3,
    )

    expected = torch.tensor(
        [
            [True, True, False, False, True, True, False, False],
            [True, True, False, False, True, True, False, False],
            [True, True, False, False, True, True, False, False],
        ]
    )
    torch.testing.assert_close(mask, expected)


def test_ee_target_mask_all_invalid_is_safe():
    mask = build_ee_target_token_mask(
        torch.zeros((2, 4), dtype=torch.bool),
        sequence_len=4,
        batch_size=1,
    )

    assert mask.shape == (1, 16)
    assert not mask.any()


def test_ee_target_mask_accepts_single_unbatched_arm_level_values():
    mask = build_ee_target_token_mask(
        torch.tensor([True, False]),
        sequence_len=3,
        batch_size=1,
    )

    expected = torch.tensor(
        [[True, True, False, False, True, True, False, False, True, True, False, False]]
    )
    torch.testing.assert_close(mask, expected)
