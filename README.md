# Pose Partition Networks for Multi-Person Pose Estimation
This repository contains the code and pretrained models of:

Xuecheng Nie, Jiashi Feng, Junliang Xing, and Shuicheng Yan. "Pose Partition Networks for Multi-Person Pose Estimation" [[ArXiv](https://arxiv.org/abs/1705.07422)].

## Prerequisites
- Python 3.5
- Pytorch 0.2.0
- OpenCV 3.0 or higher

## Installation
1. Install Pytorch: Please follow the [official instruction](https://pytorch.org/) on installation of Pytorch.
2. Clone the repository 
   ```
   git clone --recursive 
   ```
3. Download [MPII Multi-Person Human Pose](http://human-pose.mpi-inf.mpg.de/) dataset and create a symbolic link to the ```images``` directory 
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR dataset/mpi/images
   ```

## Usage

### Training

### Testing

## Citation
If you use our code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{nie2018ppn,
  title={Pose Partition Networks for Multi-Person Pose Estimation},
  author={Nie, Xuecheng and Feng, Jiashi Feng and Xing, Junliang and Yan, Shuicheng},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018}
}
```