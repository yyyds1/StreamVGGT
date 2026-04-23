from dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset


class MultiVGARobotwinDataset(MultiLatentLeRobotDataset):
    """VGA dataset interface.

    This class currently reuses the latent robotwin loader and keeps output keys
    compatible with existing action training. It is intended to be extended with
    geometry supervision fields (camera/depth) when the new VGA dataset is ready.
    """

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # Reserved optional supervision fields for VGA multi-task training.
        # Populate these in the upcoming dedicated VGA robotwin loader.
        # sample["camera_pose_gt"] = ...
        # sample["depth_gt"] = ...
        # sample["depth_valid_mask"] = ...
        return sample
