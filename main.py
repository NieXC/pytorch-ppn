import argparse
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
import json
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from nets.pose_partition_network import PPN_with_HG_MSRAInit 
from utils.data_loader import PPN_MPPE_MPI_Dataset
from utils.calc_mAP import calc_mAP
import utils.eval_util as eval_util

parser = argparse.ArgumentParser(description='PyTorch PPN for Multi-Person Pose Estimation')
parser.add_argument('--train-data', default='dataset/mpi/images', metavar='DIR', help='path to training dataset')
parser.add_argument('--train-anno', default='dataset/mpi/jsons/MPI_MP_TRAIN_annotations.json', type=str, metavar='PATH', help='path to training annotations')
parser.add_argument('--eval-data', default='dataset/mpi/images', metavar='DIR', help='path to validation or testing dataset')
parser.add_argument('--eval-anno', default='dataset/mpi/jsons/MPI_MP_VAL_annotations.json', type=str, metavar='PATH', help='path to validation or testing annotations')

parser.add_argument('-b', '--batch-size', default=20, type=int, metavar='N', help='mini-batch size (default: 20)')
parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--epochs', default=250, type=int, metavar='N', help='number of total epochs to run (default: 250)')
parser.add_argument('--snapshot-fname-prefix', default='exps/snapshots/ppn', metavar='PATH', help='path to snapshot')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers (default: 8)')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument('--evaluate', default=False, type=bool, metavar='BOOL', help='evaluate or train')
parser.add_argument('--calc-map', default=False, type=bool, metavar='BOOL', help='Calculate mAP or not')
parser.add_argument('--pred-path', default='exps/preds/mat_results/pred_keypoints_mpii_multi.mat', type=str, metavar='PATH', help='path to save the predction results in .mat format')
parser.add_argument('--visualization', default=False, type=bool, metavar='BOOL', help='visualize prediction or not')
parser.add_argument('--vis-dir', default='exps/preds/vis_results', metavar='DIR', help='path to save visualization results')

best_map = 0
map_avg_list = []
map_all_list = []

def main():
    # Global variables
    global args, best_map, map_avg_list, map_all_list
    args = parser.parse_args()

    # Welcome msg
    phase_str = '[Train and Val Phase]'
    if args.evaluate:
        phase_str = '[Testing Phase]'
    print('Pose Partition Network for Multi-Person Pose Estimation {0}'.format(phase_str))

    # Create PPN
    PPN = PPN_with_HG_MSRAInit() 
    PPN = nn.DataParallel(PPN).cuda()

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('Loading checkpoints --> {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            PPN.load_state_dict(checkpoint['state_dict'])
            args.start_epoch = checkpoint['epoch']
            best_map = checkpoint['best_map']
            map_avg_list = checkpoint['map_avg_list']
            map_all_list = checkpoint['map_all_list']
        else:
            print('No checkpoint found at --> {0}'.format(args.resume))

    PPN_params = PPN.parameters()
    cudnn.benchmark = True

    # Snapshot file names
    snapshot_fname = '{0}.pth.tar'.format(args.snapshot_fname_prefix)
    snapshot_best_fname = '{0}_best.pth.tar'.format(args.snapshot_fname_prefix)

    # Image normalization
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1])

    # Load training data
    train_loader = torch.utils.data.DataLoader(PPN_MPPE_MPI_Dataset(args.train_data, \
                                                                    args.train_anno, \
                                                                    transforms.Compose([transforms.ToTensor(), normalize,]), \
                                                                    stride=4, \
                                                                    sigma=7, \
                                                                    crop_size=256, \
                                                                    target_dist=1.171, scale_min=0.7, scale_max=1.3, \
                                                                    max_rotate_degree=40, \
                                                                    max_center_trans=40, \
                                                                    flip_prob=0.5, \
                                                                    is_visualization=False), \
                                                                    batch_size=args.batch_size, shuffle=True, \
                                                                    num_workers=args.workers, pin_memory=True)

    # Load validation data
    print('Loading validation json file: {0}...'.format(args.eval_anno))
    eval_list = []
    with open(args.eval_anno) as data_file:
        data_this = json.load(data_file)
        data_this = data_this['root']
        eval_list = eval_list + data_this
    eval_im_name_list = []
    eval_objpos_list = []
    eval_scale_provided_list = []
    eval_center_box_list = []
    for ii in range(0, len(eval_list)):
        eval_item = eval_list[ii]
        eval_im_name_list.append(eval_item['im_name'])
        eval_objpos_list.append(eval_item['group_center'])
        eval_scale_provided_list.append(eval_item['group_scale'])
        eval_center_box_list.append(eval_item['center_bbox'])
    print('Finished loading evaluation json file')

    # MSE Loss function for confidence and orientation supervision
    pose_criterion = nn.MSELoss().cuda()
    orie_criterion = nn.SmoothL1Loss().cuda()

    # RMSProp as the optimizer
    optimizer = torch.optim.RMSprop(PPN_params, args.lr)

    # Testing
    if args.evaluate == True:
        evaluate(PPN, args.eval_data, \
                      eval_im_name_list, \
                      eval_objpos_list,\
                      eval_scale_provided_list, \
                      eval_center_box_list, \
                      eval_center_box_extend_pixels=50, \
                      transform=transforms.Compose([transforms.ToTensor(), normalize,]), \
                      crop_size=384, \
                      scale_multiplier=[0.7, 0.85, 1, 1.15, 1.3], \
                      conf_th=0.1, \
                      dist_th=120, \
                      visualization=args.visualization, \
                      vis_result_dir=args.vis_dir, \
                      pred_path=args.pred_path, \
                      is_calc_map=args.calc_map)
        return
    
    for epoch in range(args.start_epoch, args.epochs):

        # Training
        train(train_loader, PPN, pose_criterion, orie_criterion, optimizer, epoch)

        torch.save({ 
                'epoch': epoch + 1,
                'state_dict': PPN.state_dict(),
                'best_map': best_map,
                'map_avg_list': map_avg_list,
                'map_all_list': map_all_list,
            }, snapshot_fname)

        # Validation
        if epoch < 100:
            val_freq = 5
        elif epoch < 200:
            val_freq = 2
        else:
            val_freq = 1
        
        if (epoch + 1) % val_freq == 0:
            map_avg = evaluate(PPN, args.eval_data, \
                                    eval_im_name_list, \
                                    eval_objpos_list, \
                                    eval_scale_provided_list, \
                                    eval_center_box_list, \
                                    eval_center_box_extend_pixels=50, \
                                    transform=transforms.Compose([transforms.ToTensor(), normalize,]), \
                                    crop_size=384, \
                                    scale_multiplier=[0.7, 0.85, 1, 1.15, 1.3], \
                                    conf_th=0.1, \
                                    dist_th=120, \
                                    visualization=args.visualization, \
                                    vis_result_dir=args.vis_dir, \
                                    pred_path=args.pred_path, \
                                    is_calc_map=True)

            is_best = map_avg > best_map
            best_map = max(map_avg, best_map)
            
            torch.save({ 
                'epoch': epoch + 1,
                'state_dict': PPN.state_dict(),
                'best_map': best_map,
                'map_avg_list': map_avg_list,
                'map_all_list': map_all_list,
            }, snapshot_fname)
            if is_best:
                shutil.copyfile(snapshot_fname,snapshot_best_fname)

# Training
def train(train_loader, model, pose_criterion, orie_criterion, optimizer, epoch):

    cur_lr = adjust_learning_rate(optimizer, epoch)

    losses = AverageMeter()
    cost_time = AverageMeter()
    train_acc = AverageMeter()

    model.train()
    
    iter_start_time = time.time()
    for i, (im, conf_target, orie_target, orie_target_weight) in enumerate(train_loader):

        im = im.cuda(async=True)
        conf_target = conf_target.float().cuda(async=True)
        orie_target = orie_target.float().cuda(async=True)
        orie_target_weight = orie_target_weight.float().cuda(async=True)

        input_var = torch.autograd.Variable(im)
        conf_target_var = torch.autograd.Variable(conf_target)
        orie_target_var = torch.autograd.Variable(orie_target)
        orie_target_weight_var = torch.autograd.Variable(orie_target_weight)
        orie_target_var = torch.mul(orie_target_var, orie_target_weight_var)        

        conf_output_list, orie_output_list = model(input_var)

        conf_loss = pose_criterion(conf_output_list[0], conf_target_var)
        orie_output = orie_output_list[0]
        orie_output = torch.mul(orie_output, orie_target_weight_var)
        orie_loss = orie_criterion(orie_output, orie_target_var)
        total_loss = conf_loss + orie_loss
        
        for s in range(1, len(conf_output_list)):
            conf_loss = pose_criterion(conf_output_list[s], conf_target_var)
            orie_output = orie_output_list[s]
            orie_output = torch.mul(orie_output, orie_target_weight_var)
            orie_loss = orie_criterion(orie_output, orie_target_var)
            total_loss = total_loss + conf_loss + orie_loss

        losses.update(total_loss.data[0], im.size(0))

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        cost_time.update(time.time() - iter_start_time)
        iter_start_time = time.time()

        if i == 0 or (i + 1) % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}] \t'
              'CurLR: {3} \t' 
              'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
              'BatchTime {cost_time.val:.3f} ({cost_time.avg:.3f}) \t'.format(
              epoch + 1, i + 1, len(train_loader), 
              cur_lr, 
              loss=losses, 
              cost_time=cost_time))

# Validation
def evaluate(model, \
             eval_im_root_dir, \
             eval_im_name_list, \
             eval_objpos_list, \
             eval_scale_provided_list, \
             eval_center_box_list, \
             eval_center_box_extend_pixels=50, \
             transform=None, \
             stride=4, \
             crop_size=256, \
             training_crop_size=256, \
             scale_multiplier=[1], \
             num_of_joints=16, \
             conf_th=0.1, \
             dist_th=120, \
             visualization=False, \
             vis_result_dir='preds/vis_results', \
             gt_path='dataset/mpi/val_gt/mpi_val_groundtruth.mat', \
             pred_path='exps/preds/mat_results/pred_keypoints_mpii_multi.mat', \
             is_calc_map=True):

    model.eval()
    mp_pose_list  = eval_util.multi_image_testing_on_mpi_mp_dataset(model, \
                                                                    eval_im_root_dir, \
                                                                    eval_im_name_list, \
                                                                    eval_objpos_list, \
                                                                    eval_scale_provided_list, \
                                                                    eval_center_box_list, \
                                                                    center_box_extend_pixels=eval_center_box_extend_pixels, \
                                                                    transform=transform, \
                                                                    stride=stride, \
                                                                    crop_size=crop_size, \
                                                                    training_crop_size=training_crop_size, \
                                                                    scale_multiplier=scale_multiplier, \
                                                                    num_of_joints=num_of_joints, \
                                                                    conf_th=conf_th, \
                                                                    dist_th=dist_th, \
                                                                    visualization=visualization, \
                                                                    vis_result_dir=vis_result_dir)

    eval_util.save_mppe_results_to_mpi_format(mp_pose_list, save_path=pred_path)
    
    map_avg = 0.0
    if is_calc_map:
        map_all = calc_mAP(gt_path=gt_path, pred_path=pred_path)
        map_avg = map_all[-1]
        map_all_list.append(map_all)    
        map_avg_list.append(map_avg)
    
    return map_avg

# Computes and stores the average and current value
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# Adjust learning rate
def adjust_learning_rate(optimizer, epoch):

    decay = 0
    if epoch + 1 >= 200:
        decay = 0.05
    elif epoch + 1 >= 170:
        decay = 0.1
    elif epoch + 1 >= 150:
        decay = 0.25
    elif epoch + 1 >= 100:
        decay = 0.5
    else:
        decay = 1

    lr = args.lr * decay

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

if __name__ == '__main__':
    main()


