import torch

from vga.se3_guidance_tokenizer import SE3GuidanceTokenizer


def _guidance(rotation=None, duration=4):
    batch, count = 1, 2
    left_pose = torch.zeros(batch, count, 7)
    right_pose = torch.zeros(batch, count, 7)
    left_pose[..., 3] = 1.0
    right_pose[..., 3] = 1.0
    if rotation is not None:
        left_pose[0, 0, 3:7] = torch.tensor(rotation)
    current_ee = torch.zeros(batch, 2, 8)
    current_ee[..., 3] = 1.0
    return {
        "left_pose_7d": left_pose,
        "right_pose_7d": right_pose,
        "left_gripper": torch.ones(batch, count),
        "right_gripper": torch.ones(batch, count),
        "duration": torch.full((batch, count), float(duration)),
        "left_valid": torch.tensor([[True, False]]),
        "right_valid": torch.tensor([[True, True]]),
        "block_valid": torch.tensor([[True, True]]),
        "current_ee": current_ee,
    }


def test_seven_tokens_per_waypoint_and_arm_specific_mask():
    tokenizer = SE3GuidanceTokenizer(embed_dim=16, max_duration=32)
    tokens, mask = tokenizer(_guidance())
    assert tokens.shape == (1, 14, 16)
    assert mask.shape == (1, 14)
    assert mask[0, :7].tolist() == [True, True, True, True, True, True, True]
    assert mask[0, 7:14].tolist() == [False, False, False, True, True, True, True]


def test_rotation_and_duration_have_separate_tokens():
    tokenizer = SE3GuidanceTokenizer(embed_dim=16, max_duration=32)
    baseline, _ = tokenizer(_guidance())
    rotated, _ = tokenizer(_guidance(rotation=[0.7071068, 0.7071068, 0.0, 0.0]))
    changed_duration, _ = tokenizer(_guidance(duration=16))
    assert not torch.allclose(baseline[:, 1], rotated[:, 1])
    assert torch.allclose(baseline[:, 0], rotated[:, 0])
    assert torch.allclose(baseline[:, 2], rotated[:, 2])
    assert not torch.allclose(baseline[:, 6], changed_duration[:, 6])
    assert torch.allclose(baseline[:, :6], changed_duration[:, :6])
    assert torch.allclose(baseline[:, 7:13], changed_duration[:, 7:13])


def test_quaternion_sign_flip_has_identical_rotation_encoding():
    tokenizer = SE3GuidanceTokenizer(embed_dim=16, max_duration=32)
    positive, _ = tokenizer(_guidance(rotation=[0.5, 0.5, 0.5, 0.5]))
    negative, _ = tokenizer(_guidance(rotation=[-0.5, -0.5, -0.5, -0.5]))
    assert torch.allclose(positive[:, 1], negative[:, 1], atol=1e-6)
