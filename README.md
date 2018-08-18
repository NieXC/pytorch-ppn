# Pose Partition Networks for Multi-Person Pose Estimation

> **[Learning Feature Pyramids for Human Pose Estimation](https://arxiv.org/abs/1708.01101)**  
> Wei Yang, Shuang Li, Wanli Ouyang, Hongsheng Li, Xiaogang Wang  
> ICCV, 2017 

This repository contains the code and pretrained models of
> **Pose Partition Networks for Multi-Person Pose Estimation**
> **Xuecheng Nie**, Jiashi Feng, Junliang Xing, and Shuicheng Yan
> European Conference on Computer Vision (ECCV), 2018
> [[ArXiv](https://arxiv.org/abs/1705.07422)].

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
3. Download [MPII Multi-Person Human Pose](http://human-pose.mpi-inf.mpg.de/) dataset and create a symbolic link to the `images` directory
   ```
   ln -s PATH_TO_MPII_IMAGES_DIR dataset/mpi/images
   ```

## Usage

### Training
Run the following command to train PPN from scratch with 8-stack of Hourglass network as backbone:
```
sh run_train.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0,1 python main.py
```

A simple way to record the training log by adding the following command:
```
2>&1 | tee exps/logs/ppn.log
```

Some configurable hyperparameters in training phase:

- `-b` mini-batch size
- `--lr` initial learning rate
- `--epochs` total number of epochs for training
- `--snapshot-fname-prefix` prefix of file name for snapshot, e.g. if set '--snapshot-fname-prefix exps/snapshots/ppn', then 'ppn.pth.tar' (latest model) and 'ppn_best.pth.tar' (model with best validation accuracy) will be generated in the folder 'exps/snapshots' 
- `--resume` path to the model for recovering training
- `-j` number of workers for loading data
- `--print-freq` print frequency

### Testing
Run the following command to evaluate PPN on MPII `validation set`:
```
sh run_test.sh
```
or 
```
CUDA_VISIBLE_DEVICES=0 python main.py --evaluate True --resume exps/snapshots/ppn_best.pth.tar
```

Run the following command to evaluate PPN on MPII `testing set`:
```
CUDA_VISIBLE_DEVICES=0 python main.py --evaluate True --resume exps/snapshots/ppn_best.pth.tar --eval-anno dataset/mpi/jsons/MPI_MP_TEST_annotations.json
```

In particular, results will be saved as a `.mat` file followed the official evaluation format of MPII Multi-Person Human Pose.

Some configurable hyperparameters in testing phase:

- `--evaluate` True for testing and false for training
- `--resume` path to the model for evaluation
- `--pred-path` path to the mat file for saving the evaluation results
- `--visualization` visualize evaluation or not
- `--vis-dir` directory for saving the visualization results

Pretrained model and validation accuracy with this code:

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Avg. | Pretrained Model |
|--------|------|----------|-------|-------|-----|------|-------|------|------------------|
|  PPN   |      |          |       |       |     |      |       |      |   [GoogleDrive]  |

*The Single-Person pose estimation model to refine Multi-Person pose estimation results will be released soon.

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
