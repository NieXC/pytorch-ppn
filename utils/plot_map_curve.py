import matplotlib
matplotlib.use('Agg')
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from utils.calc_mAP import genTableAP

path = 'exps/snapshots/ppn.pth.tar'

checkpoint = torch.load(path)

map_all_list = checkpoint['map_all_list']

for ei in range(0, len(map_all_list)):
	print('Epoch: [{0}] =============='.format(ei))
	map_all = map_all_list[ei]
	genTableAP(map_all, 'PPN')

print('Current epoch: {0}'.format(checkpoint['epoch']))
