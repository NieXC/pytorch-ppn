import os
import sys
import numpy as np
import random
import cv2

def augmentation_scale(im, scale_self, target_dist=1.171, scale_min=0.8, scale_max=1.5):
	#print "Data augmentation -- Scale"
	
	dice = random.random()
	scale_multiplier = (scale_max - scale_min) * dice + scale_min
	if scale_self == 0:
		scale_self = 1
	scale_abs = target_dist / scale_self
	scale = scale_abs * scale_multiplier

	resized_im = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

	return resized_im, scale

def augmentation_rotate(im, max_rotate_degree=40):
	#print "Data augmentation -- Rotate"

	dice = random.random()
	degree = (dice - 0.5) * 2 * max_rotate_degree

	im_width = im.shape[1]
	im_height = im.shape[0]
	M = cv2.getRotationMatrix2D(center=(im_width / 2, im_height / 2), angle=degree, scale=1)
	r = np.deg2rad(degree)
	new_im_width = abs(np.sin(r) * im_height) + abs(np.cos(r) * im_width)
	new_im_height =  abs(np.sin(r) * im_width) + abs(np.cos(r) * im_height)
	tx = (new_im_width - im_width) / 2
	ty = (new_im_height - im_height) / 2
	M[0, 2] += tx
	M[1, 2] += ty

	rotated_im = cv2.warpAffine(im, M, dsize=(int(new_im_width), int(new_im_height)), 
		                               flags=cv2.INTER_CUBIC, 
		                               borderMode=cv2.BORDER_CONSTANT, 
		                               borderValue=(128, 128, 128))

	return rotated_im, M

def augmentation_cropped(im, obj_center, crop_x=368, crop_y=368, max_center_trans=40):
	#print "Data augmentation -- Crop"

    dice_x = random.random()
    dice_y = random.random()

    x_offset = int((dice_x - 0.5) * 2 * max_center_trans)
    y_offset = int((dice_y - 0.5) * 2 * max_center_trans)

    new_obj_center_x = obj_center[0, 0] + x_offset
    new_obj_center_y = obj_center[0, 1] + y_offset

    cropped_im = np.zeros((crop_y, crop_x, 3), dtype="float") + 128 

    offset_start_x = int(new_obj_center_x - crop_x / 2.0)
    offset_start_y = int(new_obj_center_y - crop_y / 2.0)
	
    crop_start_x = max(offset_start_x, 0)
    crop_start_y = max(offset_start_y, 0)
	
    store_start_x = max(-offset_start_x, 0)
    store_start_y = max(-offset_start_y, 0)
	
    offset_end_x = int(new_obj_center_x + crop_x / 2.0)
    offset_end_y = int(new_obj_center_y + crop_y / 2.0)
	
    crop_end_x = min(offset_end_x, im.shape[1] - 1)
    crop_end_y = min(offset_end_y, im.shape[0] - 1)
	
    store_end_x = store_start_x + (crop_end_x - crop_start_x)
    store_end_y = store_start_y + (crop_end_y - crop_start_y)

    cropped_im[store_start_y:store_end_y, store_start_x:store_end_x, :] = im[crop_start_y:crop_end_y, crop_start_x:crop_end_x, :]

    return cropped_im, np.array([[crop_start_x, crop_start_y, store_start_x, store_start_y, crop_end_x, crop_end_y, store_end_x, store_end_y]])

def augmentation_flip(im, flip_prob=0.5):
	#print "Data augmentation -- Flip"

	dice = random.random()

	doflip = False
	if dice >= flip_prob:
		doflip = True

	if doflip:
		flipped_im = cv2.flip(im, 1)
	else:
		flipped_im = im.copy()

	return flipped_im, doflip

if __name__ == "__main__":
	print("Data Augmentation -- Main")

	
