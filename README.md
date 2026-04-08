# ActionVGGT

## Overview

A Vision-Action Foundation Model based on [StreamVGGT](https://wzzheng.net/StreamVGGT/) visoin backbone and [RDT2](https://rdt-robotics.github.io/rdt2/) action expert.

## Installation

1. Clone ActionVGGT
```bash
git clone https://github.com/yyyds1/StreamVGGT.git
cd StreamVGGT
```
2. Create conda environment
```bash
conda create -n ActionVGGT python=3.11 cmake=3.14.0
conda activate ActionVGGT 
```

3. Install requirements
```bash
pip install -r requirements.txt
conda install 'llvm-openmp<16'
```

### Download Pretrained Checkpoints

1. StreamVGGT: 
 - [Hugging Face](https://huggingface.co/lch01/StreamVGGT/)
 - [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/)
 - nas: `/mnt/nas/share/home/yds/actionvggt.pth`

2. RDT:
 - [Hugging Face](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)
 - nas: `/mnt/nas/share/home/yds/RDT.pth`

## Data Preparation
### Training Datasets
Use robotwin dataset in LeRobot dataset format as training dataset, which is provide by [Lingbot-VA](https://technology.robbyant.com/lingbot-va).

To download the dataset:
```bash
huggingface-cli download --repo-type dataset robbyant/robotwin-clean-and-aug-lerobot --local-dir /path/to/your/dataset
```

The dataset is also available in nas: `/mnt/nas/datasets5/robotwin_lerobot`


## Folder Structure
The overall folder structure should be organized as follows：
```
StreamVGGT
├── ckpt/
├── data/
└── src/
    ├── actionvggt/ # model definition of vision backbone
    |   ├──...
    ├── rdt/ # model definition of action expert
    |   ├──...
    └── train_va.py # training script
```

## Training

To start training, run:
```bash
NGPU=8 bash scripts/run_va_posttrain.sh
```


## Evaluation on RoboTwin-2.0

**Preparing the Environment**

You can follow the official instructions from the original RoboTwin-2.0 repository:  
[https://robotwin-platform.github.io/doc/usage/robotwin-install.html](https://robotwin-platform.github.io/doc/usage/robotwin-install.html)


In summary:

1. 
```bash
sudo apt install libvulkan1 mesa-vulkan-drivers vulkan-tools
```

2. 
```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git && cd RoboTwin && git checkout 2eeec322
```

3. modify script/requirements.txt 
```bash
transforms3d==0.4.2
sapien==3.0.0b1
scipy==1.10.1
mplib==0.2.1
gymnasium==0.29.1
trimesh==4.4.3
open3d==0.18.0
imageio==2.34.2
pydantic
zarr
openai
huggingface_hub==0.36.2
h5py
# For Description Generation
azure==4.0.0
azure-ai-inference
pyglet<2
wandb
moviepy
imageio
termcolor
av
matplotlib
ffmpeg
```

4. modify line 8 of script/_install.sh:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" --no-build-isolation
```

5. run:
```bash
bash script/_install.sh
```

6. run:
```bash
bash script/_download_assets.sh
```

 **Deploying the Inference Server**
```bash
# single GPU
bash scripts/launch_server.sh

# multi-GPU
bash scripts/launch_server_multigpus.sh
```

 **Executing the Inference Client**
```bash
# single GPU
task_name="adjust_bottle";
save_root="results/";
bash scripts/launch_client.sh ${save_root} ${task_name}

# multi-GPU
save_root="results/"
task_group_id=0;
bash scripts/launch_client_multigpus.sh ${save_root} ${task_group_id}
```

Related experiments results will be save in `/path/to/your/RoboTwin/${save_root}`. Please note that an `eval_result` folder is also generated. This is a native output from RoboTwin and is identical to the contents in the results folder; it can be safely ignored.
It is important to note that the inference server and client must be deployed on the same machine. For launching multi-GPU client, we padded the original 50 tasks to 56 via duplication and partitioned them into 7 groups to align with the 8-GPU configuration of our inference node. You can specify the `task_group_id` (0-6) to select a particular group for inference. For detailed grouping configurations, please refer to `evaluation/robotwin/launch_client_multigpus.sh`.

## Trouble Shooting

1. To load the dataset offline, set env variable:
``` bash
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
```

2. 
