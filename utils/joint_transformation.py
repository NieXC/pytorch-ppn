import os
import sys
import numpy as np
import random
import cv2

def scale_coords(joints, center, scale_param):
	#print "Data augmentation -- Scale joints"

	scaled_joints = joints * scale_param
	scaled_center = center * scale_param

	return scaled_joints, scaled_center

def rotate_coords(joints, center, rotate_param):
	#print "Data augmentation -- Rotate joints"
	joints_trans = joints.transpose()
	joints_padded = np.ones((3, joints_trans.shape[1]))
	joints_padded[0:2, :] = joints_trans
	rotated_joints = np.dot(rotate_param, joints_padded)

	center_trans = center.transpose()
	center_padded = np.ones((3, 1))
	center_padded[0:2, :] = center_trans
	rotated_center = np.dot(rotate_param, center_padded)

	return rotated_joints.transpose(), rotated_center.transpose()

def crop_coords(joints, center, crop_param):
	#print "Data augmentation -- Crop joints"

	cropped_joints = joints.copy()
	cropped_joints[:, 0] = cropped_joints[:, 0] - crop_param[0, 0] + crop_param[0, 2]
	cropped_joints[:, 1] = cropped_joints[:, 1] - crop_param[0, 1] + crop_param[0, 3]

	cropped_center = center.copy()
	cropped_center[:, 0] = cropped_center[:, 0] - crop_param[0, 0] + crop_param[0, 2]
	cropped_center[:, 1] = cropped_center[:, 1] - crop_param[0, 1] + crop_param[0, 3]

	return cropped_joints, cropped_center

def flip_coords(joints, center, flip_param, im_width):
	#print "Data augmentation -- Flip joints"

	flipped_joints = joints.copy()
	flipped_center = center.copy()
	if flip_param:	
		flipped_joints[:, 0] = im_width - 1 - flipped_joints[:, 0]
		flipped_joints = swap_left_and_right(flipped_joints)

		flipped_center[:, 0] = im_width - 1 - flipped_center[:, 0]

	return flipped_joints, flipped_center

def transform_mpi_to_ours(joints):
	# MPII R leg:  0(ankle), 1(knee), 2(hip)
	#      L leg:  5(ankle), 4(knee), 3(hip)
	#      R arms: 10(wrist), 11(elbow), 12(shoulder)
	#      L arms: 15(wrist), 14(elbow), 13(shoulder)
	#      Head:   8 - upper neck, 9 - head top
	#      Torso:  6 - pelvis, 7 - thorax

	# Ours Head: 0 - head top, 1 - upper neck
	#      R arms: 2(shoulder), 3(elbow), 4(wrist)
	#      L arms: 5(shoulder), 6(elbow), 7(wrist)
	#      R leg:  8(hip), 9(knee), 10(ankle)
	#      L leg:  11(hip), 12(knee), 13(ankle)
	#      Torso:  14 - thorax, 15 - pelvis

	mpi_to_ours = [9, 8, 12, 11, 10, 13, 14, 15, 2, 1, 0, 3, 4, 5, 7, 6]

	reordered_joints = np.zeros_like(joints)
	for ji in range(0, len(mpi_to_ours)):
		reordered_joints[ji, :] = joints[mpi_to_ours[ji], :]

	return reordered_joints

def swap_left_and_right(joints):
    # print "Data augmentation -- Swap left and right joints"

	right_idx = [2, 3, 4, 8, 9, 10]
	left_idx = [5, 6, 7, 11, 12, 13]

	swapped_joints = joints.copy()
	for i in range(0, 6):
		temp_joint = np.zeros((1, 2))
		temp_joint[0, :] = swapped_joints[right_idx[i], :]
		swapped_joints[right_idx[i], :] = swapped_joints[left_idx[i], :]
		swapped_joints[left_idx[i], :] = temp_joint[0, :]

	return swapped_joints


