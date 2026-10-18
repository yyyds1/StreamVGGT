from pathlib import Path
from unittest.mock import patch
import torch
from transformers import Dinov2Config, Dinov2Model
from vga.wrist_dinov2 import WristDINOv2Encoder, append_wrist_tokens


def _tiny_model(*args, **kwargs):
    return Dinov2Model(Dinov2Config(hidden_size=24, num_hidden_layers=1, num_attention_heads=3,
                                  intermediate_size=48, image_size=28, patch_size=14))


def test_independent_frozen_backbones_and_trainable_projection():
    with patch("vga.wrist_dinov2.AutoModel.from_pretrained", side_effect=_tiny_model):
        model = WristDINOv2Encoder("test-only-tiny", 32, image_size=28)
    model.train()
    assert not model.encoders[0].training and not model.encoders[1].training
    assert next(model.encoders[0].parameters()).data_ptr() != next(model.encoders[1].parameters()).data_ptr()
    assert all(not p.requires_grad for p in model.encoders.parameters())
    x = torch.rand(2, 2, 3, 28, 28, requires_grad=True)
    y = model(x)
    assert y.shape == (2, 8, 32)
    y.square().mean().backward()
    assert x.grad is None
    assert all(p.grad is None for p in model.encoders.parameters())
    assert all(p.grad is not None for p in model.projections.parameters())
    assert model.camera_embedding.grad is not None
    swapped = model(x.detach().flip(1))
    assert not torch.allclose(y, swapped)


def test_concat_both_condition_ranks_and_gradient():
    wrist = torch.randn(2, 8, 32, requires_grad=True)
    for head in [torch.randn(2, 4, 32), torch.randn(2, 3, 4, 32)]:
        result = append_wrist_tokens(head, wrist)
        assert result.shape[-2:] == (12, 32)
        assert torch.equal(result[..., :4, :], head)
        expected = wrist if head.ndim == 3 else wrist[:, None].expand(-1, 3, -1, -1)
        assert torch.equal(result[..., 4:, :], expected)
        result.square().mean().backward()
    assert wrist.grad is not None


def test_independent_config_keeps_ee_disabled_and_waypoint_noise_enabled():
    from configs import VA_CONFIGS
    cfg = VA_CONFIGS["head_wrist_dinov2_noisy_lora_train"]
    old = VA_CONFIGS["head_camera_sparse_7token_quat_duration_50tasks_noisy_train"]
    assert cfg.enable_wrist_dinov2 and cfg.use_lora and not cfg.freeze_visual_backbone
    assert cfg.se3_guidance_train_noise
    assert not cfg.enable_ee_target_module and not cfg.enable_ee_target_loss
    assert cfg.wrist_obs_cam_keys == ["left_camera", "right_camera"]
    assert not cfg.transformer_resume and cfg.transformer_warm_start is None
    assert cfg.save_root != old.save_root
    assert not getattr(old, "enable_wrist_dinov2", False)


def test_vga_to_rdt_selected_layers_with_wrists():
    from torch import nn
    from vga.models.vga import VGA
    from rdt.model import RDT
    class FakeAggregator(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.dim = kwargs["embed_dim"]
        def forward(self, images, **kwargs):
            b, frames = images.shape[:2]
            tokens = torch.randn(b, frames, 24, self.dim)
            return [tokens for _ in range(4)], {"image": (0,4), "ee_target": (4,6), "action": (6,22), "lang": (22,24)}
    cond={"layer_mode":"selected", "image_layers":[0,1,2,3], "action_layers":[0,1,2,3],
          "use_action_queries":True,"use_image_tokens":True,"use_language_tokens":False,"use_ee_target_tokens":True}
    with patch("vga.models.vga.VGAAggregator", FakeAggregator), patch("vga.wrist_dinov2.AutoModel.from_pretrained", side_effect=_tiny_model):
        model=VGA(embed_dim=32, window_size=8, chunk_size=9, enable_camera_depth_heads=False,
                  enable_se3_guidance=True, rdt_condition_tokens=cond, enable_wrist_dinov2=True,
                  wrist_dinov2_model="test-only-tiny", wrist_dinov2_image_size=28)
    pose=torch.zeros(1,6,7);pose[...,3]=1
    g={"left_pose_7d":pose,"right_pose_7d":pose.clone(),"left_gripper":torch.ones(1,6),
       "right_gripper":torch.ones(1,6),"duration":torch.ones(1,6),
       "left_valid":torch.ones(1,6,dtype=torch.bool),"right_valid":torch.ones(1,6,dtype=torch.bool),
       "block_valid":torch.ones(1,6,dtype=torch.bool)}
    g["current_ee"] = torch.zeros(1,2,8)
    g["current_ee"][...,3] = 1
    out=model({"image_dict":{"images":torch.rand(1,3,8,28,28),"wrist_images":torch.rand(1,2,3,28,28)},"se3_guidance":g})
    c=out.ress
    assert c["rdt_img_c"].shape == (1,4,12,32)
    assert c["rdt_act_c"].shape == (1,4,9,32)
    assert c["rdt_se3_c"].shape == (1,4,42,32)
    assert model.ee_target_head is None
    rdt_config={"hidden_size":32,"num_heads":4,"depth":12,"action_dim":14,"num_register_tokens":2,
                "norm_eps":1e-5,"multiple_of":16,"ffn_dim_multiplier":None,"num_kv_heads":2,"use_flash_attn":False}
    rdt=RDT(horizon=9,output_size=14,config=rdt_config,max_img_len=12,max_act_len=9,max_ee_len=42,
            lang_pos_emb_config=None,max_lang_len=0,dtype=torch.float32,
            img_pos_emb_config=[("image",4),("left_wrist",4),("right_wrist",4)],
            act_pos_emb_config=[("action",9)],ee_pos_emb_config=[("ee_target",42)],
            x_pos_emb_config=[("act",11)])
    pred=rdt(torch.randn(1,9,14),torch.ones(1),img_c=c["rdt_img_c"],act_c=c["rdt_act_c"],
             ee_c=c["rdt_se3_c"],ee_mask=c["rdt_se3_mask"],state_c=torch.randn(1,1,14),
             embed_input=True,decode_output=True)
    assert pred.shape==(1,9,14)
    pred.square().mean().backward()
    assert all(p.grad is None for p in model.wrist_dinov2.encoders.parameters())
    assert all(p.grad is not None for p in model.wrist_dinov2.projections.parameters())


def test_nested_lerobot_repos_are_not_silently_dropped():
    import tempfile
    from types import SimpleNamespace
    from dataset.robotwin_lerobot_dataset import MultiRobotwinLeRobotDataset
    with tempfile.TemporaryDirectory() as tmp:
        base=Path(tmp)
        for task, suffix in [("task_a", ""), ("task_b", "lerobot")]:
            meta=base/task/"demo_randomized"/"lerobot"/suffix/"meta"
            meta.mkdir(parents=True)
            (meta/"info.json").write_text("{}")
        loader=MultiRobotwinLeRobotDataset.__new__(MultiRobotwinLeRobotDataset)
        loader.dataset_path=base
        loader.config=SimpleNamespace(single_task="task_a,task_b",robotwin_lerobot_repo_subdir="demo_randomized/lerobot",
                                      require_all_requested_tasks=True)
        repos=loader._scan_repos()
        assert len(repos)==2
        assert repos[1] == base/"task_b/demo_randomized/lerobot/lerobot"
        loader.config.single_task="task_c"
        try:
            loader._scan_repos()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing requested tasks must fail instead of silently training a subset")
