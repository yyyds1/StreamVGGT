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
 -  [Hugging Face](https://huggingface.co/lch01/StreamVGGT/)
 - [Tsinghua cloud](https://cloud.tsinghua.edu.cn/d/d6ad8f36fcd541bcb246/)

2. RDT:
 - [Hugging Face](https://huggingface.co/robotics-diffusion-transformer/RDT2-FM)

## Data Preparation
### Training Datasets
The training dataset consist of six dataset in LeRobot dataset format, which is aligned with [Lingbot-VA](https://technology.robbyant.com/lingbot-va).

  - Agibot
  - RoboMind
  - InternData-A1
  - OXE
  - UMI Data
  - RoboCOIN

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

TODO


## Evaluation

TODO
