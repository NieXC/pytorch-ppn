import matplotlib
matplotlib.use('Agg')

import torch.utils.data as data
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image
import cv2
import numpy as np
import os
import os.path
import json
import random
import time

import utils.data_augmentation as data_aug
import utils.joint_transformation as joint_trans
import utils.target_generation as target_gen
import utils.vis_utils

# Use opencv to load image
def opencv_loader(path):
    return cv2.imread(path, 1)

# MPII Multi-Person dataset
class PPN_MPPE_MPI_Dataset(data.Dataset):
    def __init__(self, root, train_file, transform=None, target_transform=None, loader=opencv_loader, \
                                                                                stride=4, \
                                                                                sigma=7, \
                                                                                crop_size=256, \
                                                                                target_dist=1.171, scale_min=0.7, scale_max=1.3, \
                                                                                max_rotate_degree=40, \
                                                                                max_center_trans=40, \
                                                                                flip_prob=0.5, \
                                                                                is_visualization=False):

        # Load training json file
        print('Loading training json file: {0}...'.format(train_file))
        train_list = []
        with open(train_file) as data_file:
            data_this = json.load(data_file)
            data_this = data_this['root']
            train_list = train_list + data_this
        print('Finished loading training json file')

        # Hyper-parameters
        self.root = root
        self.train_list = train_list
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.stride = stride
        self.sigma = sigma
        self.crop_size = crop_size
        self.target_dist = target_dist
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.max_rotate_degree = max_rotate_degree
        self.max_center_trans = max_center_trans
        self.flip_prob = flip_prob

        # Number of train samples
        self.N_train = len(self.train_list)

        # Visualization or not
        self.is_visualization = is_visualization
    
    def __getitem__(self, index):

        # Select a training sample
        train_item = self.train_list[index]

        # Load training image
        im_name = train_item['img_paths']
        im = self.loader(os.path.join(self.root, im_name))

        # Get joint info
        joints_all_info = np.array(train_item['joint_self'])
        joints_loc = np.zeros((joints_all_info.shape[0], 2))
        joints_loc[:, :] = joints_all_info[:, 0:2]
        
        # Reorder joints from MPI to ours
        joints_loc = joint_trans.transform_mpi_to_ours(joints_loc)

        # Get visibility of joints (never can be seen)
        coord_sum = np.sum(joints_loc, axis=1)
        self_ori_visibility = coord_sum != 0
        self_ori_visibility = self_ori_visibility.astype(int)

        # Get person center and scale
        person_center = np.array([train_item['objpos']])
        scale_provided = train_item['scale_provided']

        # Random scaling
        scaled_im, scale_param = data_aug.augmentation_scale(im, scale_provided, target_dist=self.target_dist, scale_min=self.scale_min, scale_max=self.scale_max)
        scaled_joints, scaled_center = joint_trans.scale_coords(joints_loc, person_center, scale_param)
           
        # Random rotating
        rotated_im, rotate_param = data_aug.augmentation_rotate(scaled_im, max_rotate_degree=self.max_rotate_degree)
        rotated_joints, rotated_center = joint_trans.rotate_coords(scaled_joints, scaled_center, rotate_param)

        # Random cropping
        cropped_im, crop_param = data_aug.augmentation_cropped(rotated_im, rotated_center, crop_x=self.crop_size, crop_y=self.crop_size, max_center_trans=self.max_center_trans)
        cropped_joints, cropped_center = joint_trans.crop_coords(rotated_joints, rotated_center, crop_param)
        
        # Random flipping
        flipped_im, flip_param = data_aug.augmentation_flip(cropped_im, flip_prob=self.flip_prob)
        flipped_joints, flipped_center = joint_trans.flip_coords(cropped_joints, cropped_center, flip_param, flipped_im.shape[1])

        # If flip, then swap the visibility of left and right joints
        if flip_param:
            right_idx = [2, 3, 4, 8, 9, 10]
            left_idx = [5, 6, 7, 11, 12, 13]
            for i in range(0, 6):
                temp_visibility = self_ori_visibility[right_idx[i]]
                self_ori_visibility[right_idx[i]] = self_ori_visibility[left_idx[i]]
                self_ori_visibility[left_idx[i]] = temp_visibility

        onplane = np.logical_and(flipped_joints >= 0, flipped_joints < self.crop_size) 
        self_visibility = np.logical_and(onplane[:, 0], onplane[:, 1]).astype(int)
        self_visibility = self_ori_visibility * self_visibility

        # Generate target maps
        grid_x = flipped_im.shape[1] / self.stride
        grid_y = flipped_im.shape[0] / self.stride 

        conf_target = target_gen.gen_gaussian_maps(flipped_joints, self_visibility, self.stride, grid_x, grid_y, self.sigma)

        embedding_center = np.zeros((1, 2))
        if self_ori_visibility[15]:
            embedding_center[0, :] = flipped_joints[15, :]
        else:
            embedding_center[0, :] = flipped_center[0, :]
        
        # For recovering the points, remember that x = start + g_x * stride        
        orie_target, orie_target_weight = target_gen.gen_orientation_maps(flipped_joints, self_visibility, embedding_center, self.stride, grid_x, grid_y, self.sigma)

        # The number of other people in the image
        num_other_people = int(train_item['numOtherPeople'])

        # If there are other people in the image, then...
        if num_other_people > 0:

            # The joints of all other people
            joint_others = train_item['joint_others']
            other_joints_all_info = []
            
            # The centers and scales of other people
            other_objpos = train_item['objpos_other']
            other_objpos_list = []

            if num_other_people == 1:
                other_joints_all_info.append(np.array(joint_others)) 
                np_other_objpos = np.zeros((1, 2))
                np_other_objpos[0, :] = np.array(other_objpos)
                other_objpos_list.append(np_other_objpos)    
            else:
                for oi in range(0, num_other_people):
                    other_joints_all_info.append(np.array(joint_others[oi]))
                    np_other_objpos = np.zeros((1, 2))
                    np_other_objpos[0, :] = np.array(other_objpos[oi])
                    other_objpos_list.append(np_other_objpos)
        
            # Reorder joints of other people from MPI to ours
            other_joints_loc_list = []  
            other_ori_visibility_list = []
            for oi in range(0, num_other_people):
            
                other_joints_all_info[oi] = joint_trans.transform_mpi_to_ours(other_joints_all_info[oi])    

                # Get the joint location of other joints
                other_joints_loc = np.zeros((joints_all_info.shape[0], 2))
                other_joints_loc = other_joints_all_info[oi][:, 0:2]
                other_joints_loc_list.append(other_joints_loc)

                # Get visibility of joints (never can be seen)
                coord_sum = np.sum(other_joints_loc, axis=1)
                other_ori_visibility = coord_sum != 0
                other_ori_visibility = other_ori_visibility.astype(int)   
                other_ori_visibility_list.append(other_ori_visibility)

            # Random sacling
            scaled_other_joints_list = []
            scaled_other_objpos_list = []
            for oi in range(0, num_other_people):
                scaled_other_joints, temp_scaled_center = joint_trans.scale_coords(other_joints_loc_list[oi], person_center, scale_param)
                scaled_other_joints_list.append(scaled_other_joints)
            
                scaled_other_objpos, temp_scaled_center = joint_trans.scale_coords(other_objpos_list[oi], person_center, scale_param)
                scaled_other_objpos_list.append(scaled_other_objpos)

            # Random rotating
            rotated_other_joints_list = []
            rotated_other_objpos_list = []
            for oi in range(0, num_other_people):
                rotated_other_joints, temp_rotated_center = joint_trans.rotate_coords(scaled_other_joints_list[oi], scaled_center, rotate_param) 
                rotated_other_joints_list.append(rotated_other_joints)

                rotated_other_objpos, temp_rotated_center = joint_trans.rotate_coords(scaled_other_objpos_list[oi], scaled_center, rotate_param) 
                rotated_other_objpos_list.append(rotated_other_objpos)
                
            # Random cropping
            cropped_other_joints_list = []
            cropped_other_objpos_list = []
            for oi in range(0, num_other_people):
                cropped_other_joints, temp_cropped_center = joint_trans.crop_coords(rotated_other_joints_list[oi], rotated_center, crop_param)
                cropped_other_joints_list.append(cropped_other_joints)
            
                cropped_other_objpos, temp_cropped_center = joint_trans.crop_coords(rotated_other_objpos_list[oi], rotated_center, crop_param)
                cropped_other_objpos_list.append(cropped_other_objpos)

            # Random flipping
            flipped_other_joints_list = []
            flipped_other_objpos_list = []
            other_visibility_list = []
            for oi in range(0, num_other_people):
                flipped_other_joints, temp_flipped_center = joint_trans.flip_coords(cropped_other_joints_list[oi], cropped_center, flip_param, flipped_im.shape[1]) 
                flipped_other_joints_list.append(flipped_other_joints)

                onplane = np.logical_and(flipped_other_joints >= 0, flipped_other_joints < self.crop_size) 
                other_visibility = np.logical_and(onplane[:, 0], onplane[:, 1]).astype(int)

                other_ori_visibility = other_ori_visibility_list[oi]

                # If flip, then swap the visibility of left and right joints
                if flip_param:
                    right_idx = [2, 3, 4, 8, 9, 10]
                    left_idx = [5, 6, 7, 11, 12, 13]
                    for i in range(0, 6):
                        temp_visibility = other_ori_visibility[right_idx[i]]
                        other_ori_visibility[right_idx[i]] = other_ori_visibility[left_idx[i]]
                        other_ori_visibility[left_idx[i]] = temp_visibility

                other_visibility = other_visibility * other_ori_visibility
                other_visibility_list.append(other_visibility)

                flipped_other_objpos = cropped_other_objpos_list[oi].copy()
                if flip_param:
                    flipped_other_objpos[:, 0] = flipped_im.shape[1] - 1 - flipped_other_objpos[:, 0]
                flipped_other_objpos_list.append(flipped_other_objpos)

            # Generate target maps for other people
            for oi in range(0, num_other_people):
                other_conf_target = target_gen.gen_gaussian_maps(flipped_other_joints_list[oi], other_visibility_list[oi], self.stride, grid_x, grid_y, self.sigma)
                conf_target += other_conf_target

            conf_target[conf_target > 1] = 1 
            max_target_map = conf_target[0:joints_all_info.shape[0], :, :].max(0)
            conf_target[joints_all_info.shape[0], :, :] = 1 - max_target_map            

            other_embedding_center_list = []
            for oi in range(0, num_other_people):
                other_embedding_center = np.zeros((1, 2))
                if other_ori_visibility_list[oi][15]:
                    other_embedding_center[0, :] = flipped_other_joints_list[oi][15, :]
                else:
                    other_embedding_center[0, :] = flipped_other_objpos_list[oi][0, :]
                other_embedding_center_list.append(other_embedding_center)

            # Generate orientation target maps for all people
            flipped_other_joints_list.append(flipped_joints)
            other_visibility_list.append(self_visibility)
            other_embedding_center_list.append(embedding_center)

            orie_target, orie_target_weight = target_gen.gen_orientation_maps_from_list(flipped_other_joints_list, other_visibility_list, other_embedding_center_list, self.stride, grid_x, grid_y, self.sigma)

        # Transform
        if self.transform is not None:
            aug_im = self.transform(flipped_im)
        else:
            aug_im = flipped_im
        
        # Visualize target maps
        if self.is_visualization:
            print('Visualize joint gaussian maps and orientation maps')
            vis_utils.vis_gaussian_maps(flipped_im, conf_target, self.stride, save_im=True)
            vis_utils.vis_orientation_maps(flipped_im, orie_target, self.stride, grid_x, grid_y, save_im=True)

        return aug_im, conf_target, orie_target, orie_target_weight
    
    def __len__(self):
        return self.N_train

if __name__ == '__main__':
    print('Data loader for PPN Partition Networks on MPII Human Pose Multi-Person dataset')
