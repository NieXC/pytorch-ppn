import os
import sys
import numpy as np
import random
import cv2

def gen_single_gaussian_map(center, stride, grid_x, grid_y, sigma):
    #print "Target generation -- Single gaussian maps"

    gaussian_map = np.zeros((grid_y, grid_x))
    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    for g_y in range(start_y, end_y):
        for g_x in range(start_x, end_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            gaussian_map[g_y, g_x] += np.exp(-exponent)
            if gaussian_map[g_y, g_x] > 1:
                gaussian_map[g_y, g_x] = 1

    return gaussian_map

def gen_gaussian_maps(joints, visibility, stride=8, grid_x=46, grid_y=46, sigma=7):
    #print "Target generation -- Gaussian maps"

    joint_num = joints.shape[0]
    gaussian_maps = np.zeros((joint_num + 1, grid_y, grid_x))
    for ji in range(0, joint_num):
        if visibility[ji]:
            gaussian_map = gen_single_gaussian_map(joints[ji, :], stride, grid_x, grid_y, sigma)
            gaussian_maps[ji, :, :] = gaussian_map[:, :]

    # Get background heatmap
    max_heatmap = gaussian_maps.max(0)
	
    gaussian_maps[joint_num, :, :] = 1 - max_heatmap

    return gaussian_maps

def gen_single_orientation_map(center, objpos, stride, grid_x, grid_y, sigma):
    #print 'Target generation -- Single orientation map'

    orientation_map = np.zeros((2, grid_x, grid_y))
    orientation_weight_map = np.zeros((2, grid_x, grid_y))

    crop_size_x = stride * grid_x
    crop_size_y = stride * grid_y
    dist_x = crop_size_x / 2.0
    dist_y = crop_size_y / 2.0

    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    for g_y in range(start_y, end_y):
        for g_x in range(start_x, end_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            
            orientation_map[0, g_y, g_x] = (objpos[0, 0] - x) / dist_x
            orientation_map[1, g_y, g_x] = (objpos[0, 1] - y) / dist_y
            orientation_weight_map[0, g_y, g_x] = 1
            orientation_weight_map[1, g_y, g_x] = 1

    return orientation_map, orientation_weight_map

def gen_orientation_maps(joints, visibility, objpos, stride=8, grid_x=46, grid_y=46, sigma=7):
    #print 'Target generation -- Orientation maps'

    joint_num = joints.shape[0]
    orientation_maps = np.zeros((joint_num * 2, grid_x, grid_y))
    orientation_weight_maps = np.zeros((joint_num * 2, grid_x, grid_y))
    for ji in range(0, joint_num):
        if visibility[ji]:
            orientation_map, orientation_weight_map = gen_single_orientation_map(joints[ji, :], objpos, stride, grid_x, grid_y, sigma)
            orientation_maps[ji*2:(ji+1)*2, :, :] = orientation_map
            orientation_weight_maps[ji*2:(ji+1)*2, :, :] = orientation_weight_map

    return orientation_maps, orientation_weight_maps

def gen_single_orientation_map_from_list(orientation_maps, orientation_weight_maps, ji, center, objpos, stride, grid_x, grid_y, sigma):
    #print 'Target generation -- Single orientation map'

    crop_size_x = stride * grid_x
    crop_size_y = stride * grid_y
    dist_x = crop_size_x / 2.0
    dist_y = crop_size_y / 2.0

    start = stride / 2.0 - 0.5

    max_dist = np.ceil(np.sqrt(4.6052 * sigma * sigma * 2.0))
    start_x = int(max(0, np.floor((center[0] - max_dist - start) / stride)))
    end_x = int(min(grid_x, np.ceil((center[0] + max_dist - start) / stride)))
    start_y = int(max(0, np.floor((center[1] - max_dist - start) / stride)))
    end_y = int(min(grid_y, np.ceil((center[1] + max_dist - start) / stride)))

    for g_y in range(start_y, end_y):
        for g_x in range(start_x, end_x):
            x = start + g_x * stride
            y = start + g_y * stride
            d2 = (x - center[0]) * (x - center[0]) + (y - center[1]) * (y - center[1])
            exponent = d2 / 2.0 / sigma / sigma
            if exponent > 4.6052:
                continue
            
            orientation_maps[ji * 2, g_y, g_x] = (objpos[0, 0] - x) / dist_x
            orientation_maps[ji * 2 + 1, g_y, g_x] = (objpos[0, 1] - y) / dist_y
            orientation_weight_maps[ji * 2, g_y, g_x] = 1
            orientation_weight_maps[ji * 2 + 1, g_y, g_x] = 1

def gen_orientation_maps_from_list(joints_list, visibility_list, objpos_list, stride=8, grid_x=46, grid_y=46, sigma=7):
    #print 'Target generation -- Orientation maps from list'

    joint_num = joints_list[-1].shape[0]    
    orientation_maps = np.zeros((joint_num * 2, grid_x, grid_y))
    orientation_weight_maps = np.zeros((joint_num * 2, grid_x, grid_y))
    num = len(joints_list)
    for li in range(0, num):
        joints = joints_list[li]
        visibility = visibility_list[li]
        objpos = objpos_list[li]

        for ji in range(0, joint_num):
            if visibility[ji]:
                gen_single_orientation_map_from_list(orientation_maps, orientation_weight_maps, ji, joints[ji, :], objpos, stride, grid_x, grid_y, sigma)
                
    return orientation_maps, orientation_weight_maps

if __name__ == '__main__':
    print('Target generation')
