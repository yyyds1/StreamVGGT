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

### Evaluation Datasets

TODO

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


## Evaluation

TODO

## Trouble Shooting

1. To load the dataset offline, set env variable:
``` bash
export HF_HUB_OFFLINE=1 
export HF_DATASETS_OFFLINE=1 
export TRANSFORMERS_OFFLINE=1
```

2. 
