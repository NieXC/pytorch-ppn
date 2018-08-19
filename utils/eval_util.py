import argparse
import os
import sys
import shutil
import time
import numpy as np
from PIL import Image
import json
import cv2
from scipy.ndimage.filters import gaussian_filter
import csv
import scipy.cluster.hierarchy as hcluster
import math
from numpy.core.records import fromarrays
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils.data_augmentation import augmentation_cropped
from utils.vis_utils import vis_mppe_results

# Multi-scale Multi-staget testing for MPI Multi-Person dataset
def multi_image_testing_on_mpi_mp_dataset(net, \
                                          im_root_dir,\
                                          im_name_list, \
                                          objpos_list=None, \
                                          scale_provided_list=None, \
                                          center_box_list=None, \
                                          center_box_extend_pixels=50, \
                                          transform=None, \
                                          stride=4, \
                                          crop_size=256, \
                                          training_crop_size=256, \
                                          scale_multiplier=[1], \
                                          num_of_joints=16, \
                                          conf_th=0.1, \
                                          dist_th=120, \
                                          visualization=False, \
                                          vis_result_dir='preds/vis_results'):

    print('MPI-MP Testing with flipping multi-scale: {0} scales'.format(len(scale_multiplier)))
    num_of_im = len(im_name_list)

    total_time = 0

    mp_pose_list = [[] for i in range(num_of_im)]
    for ii in range(0, num_of_im):

        im_name = im_name_list[ii]
        im_path = os.path.join(im_root_dir, im_name)
         
        vis_im_name = 'im_{0}_mppe_vis_result.jpg'.format(ii)
        vis_im_path = os.path.join(vis_result_dir, vis_im_name)

        start_time = time.time()
        im = cv2.imread(im_path, 1)
        if objpos_list != None and scale_provided_list != None and center_box_list != None:
            objpos = objpos_list[ii]
            scale_provided = scale_provided_list[ii]
            center_box = center_box_list[ii]
        else:
            objpos = None
            scale_provided = None
            center_box = None

        mp_pose = single_image_testing_on_mpi_mp_dataset(net, im, objpos=objpos, \
                                                                  scale_provided=scale_provided, \
                                                                  center_box=center_box, \
                                                                  center_box_extend_pixels=center_box_extend_pixels, \
                                                                  transform=transform, \
                                                                  stride=stride, \
                                                                  crop_size=crop_size, \
                                                                  training_crop_size=training_crop_size, \
                                                                  scale_multiplier=scale_multiplier, \
                                                                  num_of_joints=num_of_joints, \
                                                                  conf_th=conf_th, \
                                                                  dist_th=dist_th, \
                                                                  visualization=visualization, \
                                                                  vis_im_path=vis_im_path)
        mp_pose_list[ii] = mp_pose

        end_time = time.time()
        total_time += (end_time - start_time)
         
        print('Testing for MPII Human Pose Multi-Person : [{0}/{1}], name: {2}, cur time: {3:.4f}, avg time: {4:.4f}'.format(ii + 1, num_of_im, im_name, (end_time - start_time), total_time / (ii + 1)))

    return mp_pose_list

def single_image_testing_on_mpi_mp_dataset(net, im, objpos=None, \
                                                    scale_provided=None, \
                                                    center_box=None, \
                                                    center_box_extend_pixels=50, \
                                                    transform=None, \
                                                    stride=4, \
                                                    crop_size=256, \
                                                    training_crop_size=256, \
                                                    scale_multiplier=[1], \
                                                    num_of_joints=16, \
                                                    conf_th=0.1, \
                                                    dist_th=120, \
                                                    visualization=False, \
                                                    vis_im_path='./exps/preds/vis_results/mppe_vis_result.jpg'):
    
         
    # Get the original image size
    im_height = im.shape[0]
    im_width = im.shape[1]
    long_edge = max(im_height, im_width) 

    # Get the group center
    if objpos != None and scale_provided != None and center_box != None:
        ori_center = np.array([[objpos[0], objpos[1]]])
        base_scale = 1.1714 / scale_provided
    else:
        ori_center = np.array([[im_width / 2.0, im_height / 2.0]])
        scale_provided = long_edge * 1.0 / crop_size
        base_scale = 1 / scale_provided

    # Variables to store multi-scale test images and their crop parameters
    cropped_im_list = []
    cropped_param_list = []
    flipped_cropped_im_list = []
    flipped_cropped_param_list = []

    for sm in scale_multiplier:
        # Resized image to base scales
        scale = base_scale * sm
        resized_im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_center = np.zeros([1, 2])
        scaled_center[0, 0] = int(ori_center[0, 0] * scale)
        scaled_center[0, 1] = int(ori_center[0, 1] * scale)

        # Get flipped images
        flipped_resized_im = cv2.flip(resized_im, 1)

        # Crop image for testing
        cropped_im, cropped_param = augmentation_cropped(resized_im, scaled_center, crop_x=crop_size, crop_y=crop_size, max_center_trans=0)
        cropped_im_list.append(cropped_im)
        cropped_param_list.append(cropped_param)

        scaled_flipped_center = np.zeros([1,2])
        scaled_flipped_center[0,0] = resized_im.shape[1] - scaled_center[0,0]
        scaled_flipped_center[0,1] = scaled_center[0,1]

        # Crop flipped image for testing
        flipped_cropped_im, flipped_cropped_param = augmentation_cropped(flipped_resized_im, scaled_flipped_center, crop_x=crop_size, crop_y=crop_size, max_center_trans=0)
        flipped_cropped_im_list.append(flipped_cropped_im)
        flipped_cropped_param_list.append(flipped_cropped_param)

    # Transform image
    input_im_list = []
    flipped_input_im_list = []
    if transform is not None:
        for cropped_im in cropped_im_list:
            input_im = transform(cropped_im)
            input_im_list.append(input_im)
        for flipped_cropped_im in flipped_cropped_im_list:
            flipped_input_im = transform(flipped_cropped_im)
            flipped_input_im_list.append(flipped_input_im)
    else:
        for cropped_im in cropped_im_list:
            input_im =cropped_im.copy()
            input_im_list.append(input_im)
        for flipped_cropped_im in flipped_cropped_im_list:
            flipped_input_im = flipped_cropped_im.copy()
            flipped_input_im_list.append(flipped_input_im)

    # Preparing input variable
    batch_input_im = input_im_list[0].view(-1, 3, crop_size, crop_size)
    for smi in range(1, len(input_im_list)):
        batch_input_im = torch.cat((batch_input_im, input_im_list[smi].view(-1, 3, crop_size, crop_size)), 0)
    batch_input_im = batch_input_im.cuda(async=True)
    batch_input_var = torch.autograd.Variable(batch_input_im, volatile=True)

    # Preparing flipped input variable
    batch_flipped_input_im = flipped_input_im_list[0].view(-1, 3, crop_size, crop_size)
    for smi in range(1, len(flipped_input_im_list)):
        batch_flipped_input_im = torch.cat((batch_flipped_input_im, flipped_input_im_list[smi].view(-1, 3, crop_size, crop_size)), 0)
    batch_flipped_input_im = batch_flipped_input_im.cuda(async=True)
    batch_flipped_input_var = torch.autograd.Variable(batch_flipped_input_im, volatile=True)

    # Get predicted heatmaps and convert them to numpy array
    pose_outputs, orie_outputs = net(batch_input_var)
    pose_output = pose_outputs[-1]
    pose_output = pose_output.data
    pose_output = pose_output.cpu().numpy()
    orie_output = orie_outputs[-1]
    orie_output = orie_output.data
    orie_output = orie_output.cpu().numpy()
    
    # Get predicted flipped heatmaps and convert them to numpy array
    flipped_pose_outputs, flipped_orie_outputs = net(batch_flipped_input_var)
    flipped_pose_output = flipped_pose_outputs[-1]
    flipped_pose_output = flipped_pose_output.data
    flipped_pose_output = flipped_pose_output.cpu().numpy()
    flipped_orie_output = flipped_orie_outputs[-1]
    flipped_orie_output = flipped_orie_output.data
    flipped_orie_output = flipped_orie_output.cpu().numpy()

    # First fuse the original prediction with flipped prediction
    fused_pose_output = np.zeros((pose_output.shape[0], pose_output.shape[1] - 1, crop_size, crop_size))
    flipped_idx = [0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 14, 15]
    for smi in range(0, len(scale_multiplier)):
        # Get single scale output
        single_scale_output = pose_output[smi, :, :, :].copy()
        single_scale_flipped_output = flipped_pose_output[smi, :, :, :].copy()

        # fuse each joint's heatmap
        for ji in range(0, 16):
            # Get the original heatmap
            heatmap = single_scale_output[ji, :, :].copy()
            heatmap = cv2.resize(heatmap, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

            # Get the flipped heatmap
            flipped_heatmap = single_scale_flipped_output[flipped_idx[ji], :, :].copy()
            flipped_heatmap = cv2.resize(flipped_heatmap, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)
            flipped_heatmap = cv2.flip(flipped_heatmap, 1)

            # Average the original heatmap with flipped heatmap
            heatmap += flipped_heatmap
            heatmap *= 0.5

            fused_pose_output[smi, ji, :, :] = heatmap

    # Second fuse multi-scale predictions
    base_pose_output_list = []
    base_crop_param_list = []
    for smi in range(0, len(scale_multiplier)):
        single_scale_output = fused_pose_output[smi, :, :, :]
        crop_param = cropped_param_list[smi]

        # Crop the heatmaps without padding
        cropped_single_scale_output = single_scale_output[:, crop_param[0, 3]:crop_param[0, 7], crop_param[0, 2]:crop_param[0, 6]]

        # Resize the cropped heatmaps to base scale
        cropped_single_scale_output = cropped_single_scale_output.transpose((1, 2, 0))
        base_single_scale_output = cv2.resize(cropped_single_scale_output, None, fx=1.0/scale_multiplier[smi], fy=1.0/scale_multiplier[smi], interpolation=cv2.INTER_LINEAR)
        base_single_scale_output = base_single_scale_output.transpose((2, 0, 1))

        # Resize the cropping parameters
        base_crop_param = crop_param * (1.0 / scale_multiplier[smi])

        # Add to list
        base_pose_output_list.append(base_single_scale_output)
        base_crop_param_list.append(base_crop_param)

    # Multi-scale fusion results
    ms_fused_pose_output = np.zeros((base_pose_output_list[0].shape))

    # Accumulate map for division
    accumulate_map = np.zeros((base_pose_output_list[0].shape)) + 1

    # Use the smallest image as reference
    base_start_x = int(base_crop_param_list[0][0, 0])
    base_start_y = int(base_crop_param_list[0][0, 1])
    for smi in range(0, len(scale_multiplier)):
        # Get base parameters and pose output
        base_crop_param = base_crop_param_list[smi]
        base_pose_output = base_pose_output_list[smi]

        # Temporary pose heatmaps
        temp_pose_output = np.zeros_like(ms_fused_pose_output)

        # Relative location for reference image
        store_start_x = int(base_crop_param[0, 0]) - base_start_x
        store_start_y = int(base_crop_param[0, 1]) - base_start_y
        store_end_x = int(min(store_start_x + base_pose_output.shape[2], ms_fused_pose_output.shape[2]))
        store_end_y = int(min(store_start_y + base_pose_output.shape[1], ms_fused_pose_output.shape[1]))

        temp_pose_output[:, store_start_y:store_end_y, store_start_x:store_end_x] = base_pose_output[:, 0:(store_end_y - store_start_y), 0:(store_end_x - store_start_x)]
        ms_fused_pose_output += temp_pose_output

        # Update the accumulate map
        if smi >= 1:
            accumulate_map[:, store_start_y:store_end_y, store_start_x:store_end_x] += 1
    
    # Average by the accumulate map
    # Every position should add at leat once, avoid divide by 0; also avoide dominated by center cropping
    accumulate_map[accumulate_map == 0] = len(scale_multiplier)
    ms_fused_pose_output = np.divide(ms_fused_pose_output, accumulate_map)
    
    # Get the final prediction results
    pred_joints = np.zeros((num_of_joints, 3))

    # Perform NMS to find joint candidates
    all_peaks = []
    peak_counter = 0
    for ji in range(0, num_of_joints):
        heatmap_ori = ms_fused_pose_output[ji, :, :]
        heatmap = gaussian_filter(heatmap_ori, sigma=3)

        heatmap_left = np.zeros(heatmap.shape)
        heatmap_left[1:, :] = heatmap[:-1, :]
        heatmap_right = np.zeros(heatmap.shape)
        heatmap_right[:-1, :] = heatmap[1:, :]
        heatmap_up = np.zeros(heatmap.shape)
        heatmap_up[:, 1:] = heatmap[:, :-1]
        heatmap_down = np.zeros(heatmap.shape)
        heatmap_down[:, :-1] = heatmap[:, 1:]
    
        peaks_binary = np.logical_and.reduce((heatmap >= heatmap_left, heatmap >= heatmap_right, heatmap >= heatmap_up, heatmap >= heatmap_down, heatmap > conf_th))
        peaks = list(zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]))
        peaks_with_score = [x + (heatmap_ori[x[1], x[0]], ) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i], ) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)

    # Recover the peaks to locations in original image
    cropped_param = base_crop_param_list[0]
    all_joint_candi_list = []
    for ji in range(0, len(all_peaks)):
        joint_candi_list = []
        peaks_base = all_peaks[ji]
        for ci in range(0, len(peaks_base)):
            joint_candi = np.zeros((1, 4))
            joint_candi[0, :] = np.array(peaks_base[ci])
            joint_candi[0, 0] = (joint_candi[0, 0] + cropped_param[0, 0]) / base_scale
            joint_candi[0, 1] = (joint_candi[0, 1] + cropped_param[0, 1]) / base_scale
            joint_candi_list.append(joint_candi)
        all_joint_candi_list.append(joint_candi_list)

    # Get the center embedding results
    start = stride / 2.0 - 0.5
    all_embedding_list = []
    for ji in range(0, len(all_joint_candi_list)):
        joint_candi_list = all_joint_candi_list[ji]
        embedding_list = []
        for ci in range(0, len(joint_candi_list)):
            joint_candi = joint_candi_list[ci][0, 0:2]
            offset_x_avg = 0.0
            offset_y_avg = 0.0
            valid_offset_count = 0.0
            embedding = np.zeros((1, 2))
            for si in range(0, len(scale_multiplier)):
                orie_maps = orie_output[si, :, :, :]

                flipped_orie_maps = flipped_orie_output[si, :, :, :]

                joint_candi_scaled = joint_candi * scale_multiplier[si] * base_scale
                joint_candi_scaled[0] = joint_candi_scaled[0] - cropped_param_list[si][0, 0] + cropped_param_list[si][0, 2]
                joint_candi_scaled[1] = joint_candi_scaled[1] - cropped_param_list[si][0, 1] + cropped_param_list[si][0, 3]
                g_x = int((joint_candi_scaled[0] - start) / stride)
                g_y = int((joint_candi_scaled[1] - start) / stride)
                if g_x >= 0 and g_x < crop_size / stride and g_y >= 0 and g_y < crop_size / stride:
                    offset_x = orie_maps[ji * 2, g_y, g_x]
                    offset_y = orie_maps[ji * 2 + 1, g_y, g_x]

                    flipped_offset_x = flipped_orie_maps[flipped_idx[ji] * 2, g_y, crop_size / stride - g_x - 1]
                    flipped_offset_y = flipped_orie_maps[flipped_idx[ji] * 2 + 1, g_y, crop_size / stride - g_x - 1]
                    offset_x = (offset_x - flipped_offset_x) / 2.0
                    offset_y = (offset_y + flipped_offset_y) / 2.0

                    offset_x *= training_crop_size / 2.0  
                    offset_y *= training_crop_size / 2.0 
                    offset_x = offset_x / (scale_multiplier[si] * base_scale)
                    offset_y = offset_y / (scale_multiplier[si] * base_scale)
                    offset_x_avg += offset_x
                    offset_y_avg += offset_y
                    valid_offset_count += 1

            if valid_offset_count > 0:
                offset_x_avg /= valid_offset_count
                offset_y_avg /= valid_offset_count
            embedding[0, 0] = joint_candi[0] + offset_x_avg
            embedding[0, 1] = joint_candi[1] + offset_y_avg
            embedding_list.append(embedding)
        all_embedding_list.append(embedding_list)
        
    # Convert to np array
    all_embedding_np_array = np.empty((0, 2))
    for ji in range(0, len(all_embedding_list)):
        embedding_list = all_embedding_list[ji]
        for ci in range(0, len(embedding_list)):
            embedding = embedding_list[ci]
            all_embedding_np_array = np.vstack((all_embedding_np_array, embedding))

    all_joint_candi_np_array = np.empty((0, 5))
    for ji in range(0, len(all_joint_candi_list)):
        joint_candi_list = all_joint_candi_list[ji]
        for ci in range(0, len(joint_candi_list)):
            joint_candi_with_type = np.zeros((1, 5))
            joint_candi = joint_candi_list[ci]
            joint_candi_with_type[0, 0:4] = joint_candi[0, :]
            joint_candi_with_type[0, 4] = ji
            all_joint_candi_np_array = np.vstack((all_joint_candi_np_array, joint_candi_with_type))

    # Cluster the embeddings
    if all_embedding_np_array.shape[0] < 2:
        clusters = [-1]
    else:
        Z = hcluster.linkage(all_embedding_np_array, method='centroid')
        clusters = hcluster.fcluster(Z, dist_th, criterion='distance')
        clusters = clusters - 1
    
    # Get people structure by greedy search
    num_of_people = max(clusters) + 1
    joint_idx_list = [1, 
                      0, 2, 5, 8, 11,
                         3, 6, 9, 12,
                         4, 7, 10, 13]

    people = []
    for pi in range(0, num_of_people):
        joint_of_person_idx = np.where(clusters == pi)[0]
        joint_candi_cur_persons = all_joint_candi_np_array[joint_of_person_idx, 0:3]
        end_candi_cur_persons = all_embedding_np_array[joint_of_person_idx, :]
        joint_type_cur_person = all_joint_candi_np_array[joint_of_person_idx, 4]
        if len(joint_type_cur_person) > len(np.unique(joint_type_cur_person)):
            persons = []    
            persons_ends_list = []
            for joint_idx in joint_idx_list:
                # If the joint is neck, do initialization
                if joint_idx == 1:
                    neck_candi = np.where(joint_type_cur_person == joint_idx)[0]
                    for ni in range(len(neck_candi)):
                        person = {}
                        person[str(joint_idx)] = joint_candi_cur_persons[neck_candi[ni], :]
                        persons.append(person)
                        persons_ends = np.zeros((1, 2))
                        persons_ends[0, :] = end_candi_cur_persons[neck_candi[ni], :]
                        persons_ends_list.append(persons_ends)
                # For other joints, do connection
                else:
                    other_candi = np.where(joint_type_cur_person == joint_idx)[0]
                    other_pos = end_candi_cur_persons[other_candi]
                    person_centers = np.zeros((len(persons), 2))
                    person_idx = np.zeros((len(persons), 1), dtype=np.int)
                    for mi in range(len(persons_ends_list)):
                        person_centers[mi, :] = np.mean(persons_ends_list[mi], axis=0)
                        person_idx[mi] = mi
                    while (other_candi.shape[0] > 0 and person_centers.shape[0] > 0):
                        dist_matrix = np.zeros((other_candi.shape[0], person_centers.shape[0]))
                        for hi in range(other_candi.shape[0]):
                            for ci in range(person_centers.shape[0]):
                                offset_vec = other_pos[hi, :] - person_centers[ci, :]
                                dist = math.sqrt(offset_vec[0] * offset_vec[0] + offset_vec[1] * offset_vec[1])
                                dist_matrix[hi, ci] = dist
                        connection = np.where(dist_matrix == dist_matrix.min())
                        persons[person_idx[connection[1][0], 0]][str(joint_idx)] = joint_candi_cur_persons[other_candi[connection[0][0]], :]
                        persons_ends_list[person_idx[connection[1][0], 0]] = np.vstack((persons_ends_list[person_idx[connection[1][0], 0]], 
                                                                                        end_candi_cur_persons[other_candi[connection[0][0]], :]))

                        other_candi = np.delete(other_candi, connection[0][0], axis=0)
                        other_pos = np.delete(other_pos, connection[0][0], axis=0)
                        person_centers = np.delete(person_centers, connection[1][0], axis=0)
                        person_idx = np.delete(person_idx, connection[1][0], axis=0)
                    if other_candi.shape[0] > 0 and joint_idx < 2:
                        # Add new person to list
                        for hi in range(other_candi.shape[0]):
                            person = {}
                            person[str(joint_idx)] = joint_candi_cur_persons[other_candi[hi], :]
                            persons.append(person)
                            persons_ends = np.zeros((1, 2))
                            persons_ends[0, :] = end_candi_cur_persons[other_candi[hi], :]
                            persons_ends_list.append(persons_ends)
            for person in persons:
                people.append(person)
        else:
            person = {}
            for ji in range(0, len(joint_of_person_idx)):
                person[str(int(all_joint_candi_np_array[joint_of_person_idx[ji], 4]))] = all_joint_candi_np_array[joint_of_person_idx[ji], :]
            people.append(person)

    
    if objpos != None and scale_provided != None and center_box != None:
        # Exclude out of group persons
        extend_pixels = center_box_extend_pixels
        extend_pixels = extend_pixels / base_scale
        extend_center_box = np.zeros((4, 1)) 
        extend_center_box[0] = max(0, int(center_box[0] - extend_pixels))
        extend_center_box[1] = max(0, int(center_box[1] - extend_pixels))
        extend_center_box[2] = min(im_width, int(center_box[2] + extend_pixels))
        extend_center_box[3] = min(im_height, int(center_box[3] + extend_pixels))

        num_of_people = len(people)
        center_of_mass = np.zeros((num_of_people, 2))

        for pi in range(0, num_of_people):
            person = people[pi]
            point = {}
            point['x'] = []
            point['y'] = []
            for ji in range(0, num_of_joints):
                if str(ji) in person:
                    point['x'].append(person[str(ji)][0])
                    point['y'].append(person[str(ji)][1])
            if len(point['x']) > 0 and len(point['y']) > 0:
                center_of_mass[pi, 0] = np.mean(point['x'])
                center_of_mass[pi, 1] = np.mean(point['y'])

        isInExtendedBBox = np.zeros((num_of_people, 1))
        for pi in range(0, num_of_people):
            com = center_of_mass[pi, :]
            if (com[0] >= extend_center_box[0] and com[1] >= extend_center_box[1]) and (com[0] <= extend_center_box[2] and com[1] <= extend_center_box[3]):
               isInExtendedBBox[pi] = 1

        people_in_center_box = []
        for pi in range(0, num_of_people):
            if isInExtendedBBox[pi] == 1:
                people_in_center_box.append(people[pi])
    else:
        people_in_center_box = people

    # Reture prediction results
    joint_idx_mapping = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5]
    annopoints_array = []
    
    for pi in range(0, len(people_in_center_box)):
        person = people_in_center_box[pi]
        point = {}
        point['x'] = []
        point['y'] = []
        point['score'] = []
        point['id'] = []
        for ji in range(0, 14):
            if str(ji) in person:
                point['x'].append(person[str(ji)][0])
                point['y'].append(person[str(ji)][1])
                point['score'].append(person[str(ji)][2])
                point['id'].append(joint_idx_mapping[ji])
        points_struct = fromarrays([point['x'], point['y'], point['id'], point['score']], names=['x', 'y', 'id', 'score'])
        if len(points_struct) < 4:
            continue
        annopoints = {}
        annopoints['point'] = points_struct
        annopoints_array.append(annopoints)    
    
    # If the is no detected point, add random dummy persons
    dummy_joint_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    if len(annopoints_array) == 0:
        for pi in range(0, np.random.randint(2, 5)):
            point = {}
            point['x'] = []
            point['y'] = []
            point['score'] = []
            point['id'] = []
            for ji in range(0, np.random.randint(2, len(dummy_joint_id))):
                point['x'].append(np.float64(crop_size / 2.0))
                point['y'].append(np.float64(crop_size / 2.0))
                point['score'].append(np.float64(0.5))
                point['id'].append(int(dummy_joint_id[ji]))
            points_struct = fromarrays([point['x'], point['y'], point['id'], point['score']], names=['x', 'y', 'id', 'score'])
            annopoints = {}
            annopoints['point'] = points_struct
            annopoints_array.append(annopoints)
    
    mp_pose = fromarrays([annopoints_array], names=['annopoints'])

    if visualization:
        vis_mppe_results(im, people_in_center_box, save_im=True, save_path=vis_im_path)    

    return mp_pose

def save_mppe_results_to_mpi_format(mp_pose_list, save_path='./exps/preds/mat_results/pred_keypoints_mpii_multi.mat'):
    pred = fromarrays([mp_pose_list], names=['annorect'])
    sio.savemat(save_path, {'pred': pred})

if __name__ == '__main__':
    print('Testing pose models...')
