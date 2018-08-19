import os
import sys
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt
import math

joint_names = ["Head top", "Neck",
               "R shoulder", "R elbow", "R wrist",
               "L shoulder", "L elbow", "L wrist",
               "R hip", "R knee", "R ankle",
               "L hip", "L knee", "L ankle",
               "Thorax", "Pelvis",
               "Background"]

def vis_gaussian_maps(im, gaussian_maps, stride, save_im=False, save_path='./exps/preds/vis_results/gaussian_map_on_im.jpg'):
    # print 'Visualize gaussian maps'

    gm_num = gaussian_maps.shape[0]
    plot_grid_size = np.ceil(np.sqrt(gm_num))
    for gmi in range(0, gm_num):
        gaussian_map = gaussian_maps[gmi, :, :].copy()
        if gaussian_map.max() > 0:
            gaussian_map -= gaussian_map.min()
            gaussian_map /= gaussian_map.max()
        resized_gaussian_map = gaussian_map * 255
        resized_gaussian_map = cv2.resize(resized_gaussian_map, None, fx=stride, fy=stride, interpolation=cv2.INTER_LINEAR)
        resized_gaussian_map = resized_gaussian_map.astype(np.uint8)
        resized_gaussian_map = cv2.applyColorMap(resized_gaussian_map, cv2.COLORMAP_JET)
        vis_gaussian_map_im = cv2.addWeighted(resized_gaussian_map, 0.5, im.astype(np.uint8), 0.5, 0.0);

        plt.subplot(plot_grid_size, plot_grid_size, gmi + 1), plt.imshow(vis_gaussian_map_im[:, :, [2, 1, 0]]), plt.title(joint_names[gmi], **{'size':'10'})
        plt.xticks([])
        plt.yticks([])
    
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.15)    

    if save_im:	
        plt.savefig(save_path)

def vis_orientation_maps(im, orientation_maps, stride, grid_x, grid_y, save_im=False, save_path='./exps/preds/vis_results/orientation_map_on_im.jpg'):
    # print 'Visualize orientation  maps'

    om_num = int(orientation_maps.shape[0] / 2)
    plot_grid_size = np.ceil(np.sqrt(om_num))
    start = stride / 2.0 - 0.5
    for omi in range(0, om_num):
        orientation_map = orientation_maps[omi*2:(omi+1)*2, :, :].copy()
        vis_im = im.copy().astype(np.uint8)
        for g_y in range(0, grid_y):
            for g_x in range(0, grid_x):
                x = start + g_x * stride
                y = start + g_y * stride
                offset_x = orientation_map[0, g_y, g_x]
                offset_y = orientation_map[1, g_y, g_x]
                if offset_x == 0 and offset_y == 0:
                    continue
                else:
                    cv2.line(vis_im, (int(x), int(y)), \
                                     (int(x + offset_x * stride * grid_x / 2), int(y + offset_y * stride * grid_y / 2)), \
			                         (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.circle(vis_im, (int(x), int(y)), 1, [0, 0, 255], -1, cv2.LINE_AA)           
                    cv2.circle(vis_im, (int(x + offset_x * stride * grid_x / 2), int(y + offset_y * stride * grid_y / 2)), 3, [255, 0, 0], -1, cv2.LINE_AA)
                               
        plt.subplot(plot_grid_size, plot_grid_size, omi + 1),plt.imshow(vis_im[:, :, [2, 1, 0]]), plt.title(joint_names[omi], **{'size':'10'})
        plt.xticks([])
        plt.yticks([])

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.05, right=0.95, hspace=0.35, wspace=0.15)    

    if save_im:	
        plt.savefig(save_path)

def vis_mppe_results(im, people, save_im=False, save_path='./exps/preds/vis_results/mppe_vis_result.jpg'):
    limbs = [[1, 0],
             [1, 2], [2, 3], [3, 4],
             [1, 5], [5, 6], [6, 7],
             [1, 8], [8, 9], [9, 10],
             [1, 11], [11, 12], [12, 13]]

    joint_idx_list = [1, 
                      0, 2, 5, 8, 11,
                      3, 6, 9, 12,
                      4, 7, 10, 13]

    num_of_people = len(people)
    connect_im = im.copy()

    person_colors = [[233, 161, 0],
                     [0, 193, 249],
                     [26, 16, 229],
                     [42, 113, 186], 
                     [239, 172, 0],
                     [60, 108, 47],
                     [46, 41, 166],
                     [51, 153, 255],
                     [255, 153, 102],
                     [255, 102, 102]]

    joint_colors = [[255, 0, 0], [255, 85, 0],  [255, 170, 0], [255, 255, 0], 
		            [170, 255, 0], [85, 255, 0],  [0, 255, 0],   [0, 255, 85], 
		            [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], 
		            [0, 0, 255],  [85, 0, 255], [255, 0, 170], [255, 0, 85]]

    pci = 0
    for pi in range(0, num_of_people):
        if pi > len(person_colors) - 1:
            pci = 0
        person = people[pi]
        for li in range(0, len(limbs)):
            if (str(limbs[li][0]) in person) and (str(limbs[li][1]) in person):
                point1 = person[str(limbs[li][0])]
                point2 = person[str(limbs[li][1])]  
                cv2.line(connect_im, (int(point1[0]), int(point1[1])), (int(point2[0]), int(point2[1])), person_colors[pci], 4, cv2.LINE_AA)
        pci = pci + 1
        for ji in joint_idx_list:
            if str(ji) in person:
                cv2.circle(connect_im, (int(person[str(ji)][0]), int(person[str(ji)][1])), 6, joint_colors[ji], -1, cv2.LINE_AA)
                cv2.circle(connect_im, (int(person[str(ji)][0]), int(person[str(ji)][1])), 6, [0, 0, 0], 1, cv2.LINE_AA)

    if save_im:
        cv2.imwrite(save_path, connect_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])




