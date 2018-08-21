# Pose Partition Networks for Multi-Person Pose Estimation

This repository contains the code and pretrained models of
> **Pose Partition Networks for Multi-Person Pose Estimation** [[PDF](https://niexc.github.io/assets/pdf/ppn_eccv2018.pdf)]     
> **Xuecheng Nie**, Jiashi Feng, Junliang Xing, and Shuicheng Yan   
> European Conference on Computer Vision (ECCV), 2018     

## Prerequisites

- Python 3.5
- Pytorch 0.2.0
- OpenCV 3.0 or higher

## Installation

1. Install Pytorch: Please follow the [official instruction](https://pytorch.org/) on installation of Pytorch.
2. Clone the repository   
   ```
   git clone --recursive https://github.com/NieXC/pytorch-ppn.git
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

Pretrained model and validation accuracy (measured by mAP) with this code:

| Method | Head | Shoulder | Elbow | Wrist | Hip | Knee | Ankle | Avg. | Pretrained Model |
|:------:|:----:|:--------:|:-----:|:-----:|:---:|:----:|:-----:|:----:|:----------------:|
| PPN    | 94.0 | 90.9     | 81.2  | 74.1  | 77.1| 73.4 | 67.5  | 79.7 | [GoogleDrive](https://drive.google.com/file/d/15T344y19zsmvkYMpo8NTBfYFvWaspQI7/view?usp=sharing)  |

*The [Single-Person pose estimation model](https://github.com/NieXC/pytorch-pil) to refine Multi-Person pose estimation results will be released soon.

## Citation

If you use our code/model in your work or find it is helpful, please cite the paper:
```
@inproceedings{nie2018ppn,
  title={Pose Partition Networks for Multi-Person Pose Estimation},
  author={Nie, Xuecheng and Feng, Jiashi and Xing, Junliang and Yan, Shuicheng},
  booktitle={ECCV},
  year={2018}
}
```
