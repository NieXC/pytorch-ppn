import torch
import torch.nn as nn
import math
import time
import numpy as np

from nets.network_init import GaussianInit, MSRAInit

# Pre-activation residual block
class ResidualBlock_PreAct(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ResidualBlock_PreAct, self).__init__()
		
        self.in_plane = in_plane
        self.out_plane = out_plane

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(in_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_plane, out_plane, 1, bias=False)	
        )
        self.res_block = nn.Sequential(
            nn.BatchNorm2d(in_plane),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_plane, int(out_plane / 2), 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), int(out_plane / 2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), out_plane, 1, bias=False)
    )

    def forward(self, x):

        out = self.res_block(x)

        residual = x
        if self.in_plane != self.out_plane:
	        residual = self.conv1x1(x)		
        out += residual

        return out

# Post-activation residual block
class ResidualBlock_PostAct(nn.Module):
    def __init__(self, in_plane, out_plane):
        super(ResidualBlock_PostAct, self).__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_plane, out_plane, 1, bias=False),
            nn.BatchNorm2d(out_plane)
        )
        self.res_block = nn.Sequential(
            nn.Conv2d(in_plane, int(out_plane / 2), 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), int(out_plane / 2), 3, 1, 1, bias=False),
            nn.BatchNorm2d(int(out_plane / 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_plane / 2), out_plane, 1, bias=False),
            nn.BatchNorm2d(out_plane)
        )
        self.relu = nn.ReLU(inplace=True)

        self.in_plane = in_plane
        self.out_plane = out_plane

    def forward(self, x):	

        out = self.res_block(x)

        residual = x
        if self.in_plane != self.out_plane:
            residual = self.conv1x1(x)		

        out += residual
        out = self.relu(out)

        return out

# Hourglass block
class HourglassBlock(nn.Module):
    def __init__(self, num_of_feat=256, num_of_module=1):
        super(HourglassBlock, self).__init__()
		
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.bn = nn.BatchNorm2d(num_of_feat)
        self.relu = nn.ReLU(inplace=True)

        self.srb1 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb2 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb3 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb4 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb5 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb6 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb7 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb8 = self._make_seq_res_blocks(num_of_feat, num_of_module)
        self.srb9 = self._make_seq_res_blocks(num_of_feat, num_of_module)

    # Construct sequence of residual blocks
    def _make_seq_res_blocks(self, num_of_feat, num_of_module):
        seq_res_blocks = []

        for i in range(num_of_module):
            seq_res_blocks.append(ResidualBlock_PreAct(num_of_feat, num_of_feat))

        return nn.Sequential(*seq_res_blocks)

    def forward(self, x):
        # Downsample process
        x1 = self.srb1(x)                        # 1/1
        x1_downsample = self.downsample(x1)      # 1/2
        x2 = self.srb2(x1_downsample)            # 1/2
        x2_downsample = self.downsample(x2)      # 1/4
        x3 = self.srb3(x2_downsample)            # 1/4
        x3_downsample = self.downsample(x3)      # 1/8
        x4 = self.srb4(x3_downsample)            # 1/8
        x4_downsample = self.downsample(x4)      # 1/16

        # Bottle neck
        bottle_neck = self.srb5(x4_downsample)   # 1/16

        # Upsample process
        x4_upsample = self.upsample(bottle_neck) # 1/8
        x4_sym = x4 + x4_upsample                # 1/8 
        x4_sym = self.srb6(x4_sym)               # 1/8
        x3_upsample = self.upsample(x4_sym)      # 1/4
        x3_sym = x3 + x3_upsample                # 1/4
        x3_sym = self.srb7(x3_sym)               # 1/4
        x2_upsample = self.upsample(x3_sym)      # 1/2
        x2_sym = x2 + x2_upsample                # 1/2
        x2_sym = self.srb8(x2_sym)               # 1/2
        x1_upsample = self.upsample(x2_sym)      # 1/1
        x1_sym = x1 + x1_upsample                # 1/1
        x1_sym = self.srb9(x1_sym)               # 1/1

        x1_sym = self.bn(x1_sym)
        x1_sym = self.relu(x1_sym)

        return x1_sym

# Hourglass network
class HourglassNetwork(nn.Module):
    def __init__(self, num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
        super(HourglassNetwork, self).__init__()
		
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False)
        )
		
        self.basic_block = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock_PostAct(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.num_of_stages = num_of_stages

        self.res_pre_act = ResidualBlock_PreAct(128, num_of_feat)

        self.bn = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(num_of_stages)]) 

        self.hg_list = self._make_multi_stage_hg(num_of_feat, num_of_module, num_of_stages)
        self.conv1x1_1_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages)
        self.conv1x1_2_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages - 1)
        self.conv_pred_list = self._make_multi_stage_conv1x1(num_of_feat, num_of_class, num_of_stages)
        self.conv_remap_list = self._make_multi_stage_conv1x1(num_of_class, num_of_feat, num_of_stages - 1)

        self.relu = nn.ReLU(inplace=True)

    def _make_multi_stage_hg(self, num_of_feat, num_of_module, num_of_stages):
		
        hg_list = nn.ModuleList([HourglassBlock(num_of_feat, num_of_module) for i in range(num_of_stages)])
		
        return hg_list
    
    def _make_multi_stage_conv1x1(self, in_num_of_feat, out_num_of_feat, num_of_stages):

        conv1x1_list = nn.ModuleList([nn.Conv2d(in_num_of_feat, out_num_of_feat, 1, 1) for i in range(num_of_stages)])

        return conv1x1_list

    def forward(self, x):

        pred_list = []

        pool = self.basic_block(x)
        residual = pool
        out = self.res_block(pool)
        out += residual
        feat = self.res_pre_act(out)

        feat_to_next_stage = feat
        for i in range(self.num_of_stages):
			
            hg_out = self.hg_list[i](feat_to_next_stage)
			
            feat_1 = self.conv1x1_1_list[i](hg_out)
            feat_1 = self.bn[i](feat_1)
            feat_1 = self.relu(feat_1)
			
            pred = self.conv_pred_list[i](feat_1)
            pred_list.append(pred)
			
            if i < self.num_of_stages - 1:
                feat_2 = self.conv1x1_2_list[i](feat_1)
                feat_remap = self.conv_remap_list[i](pred)
                feat_to_next_stage = feat_to_next_stage + feat_2 + feat_remap

        return pred_list

# Hourglass network for multi person pose estimation with PPN
class PPN_with_HG(nn.Module):
    def __init__(self, num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
        super(PPN_with_HG, self).__init__()
		
        self.res_block = nn.Sequential(
            nn.Conv2d(128, 64, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 1, bias=False)
        )
		
        self.basic_block = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock_PostAct(64, 128),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.num_of_stages = num_of_stages

        self.res_pre_act = ResidualBlock_PreAct(128, num_of_feat)

        self.hg_list = self._make_multi_stage_hg(num_of_feat, num_of_module, num_of_stages)

        self.relu = nn.ReLU(inplace=True)

        # For branch 1: joint detection
        self.bn_b1 = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(num_of_stages)]) 
        self.conv1x1_1_list_b1 = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages)
        self.conv1x1_2_list_b1 = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages - 1)
        self.conv_pred_list_b1 = self._make_multi_stage_conv1x1(num_of_feat, num_of_class, num_of_stages)
        self.conv_remap_list_b1 = self._make_multi_stage_conv1x1(num_of_class, num_of_feat, num_of_stages - 1)
    
        # For branch 2: dense regression
        self.bn_b2 = nn.ModuleList([nn.BatchNorm2d(num_of_feat) for i in range(num_of_stages)]) 
        self.conv1x1_1_list_b2 = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages)
        self.conv1x1_2_list_b2 = self._make_multi_stage_conv1x1(num_of_feat, num_of_feat, num_of_stages - 1)
        self.conv_pred_list_b2 = self._make_multi_stage_conv1x1(num_of_feat, (num_of_class - 1) * 2, num_of_stages)
        self.conv_remap_list_b2 = self._make_multi_stage_conv1x1((num_of_class - 1) * 2, num_of_feat, num_of_stages - 1)

    def _make_multi_stage_hg(self, num_of_feat, num_of_module, num_of_stages):
		
        hg_list = nn.ModuleList([HourglassBlock(num_of_feat, num_of_module) for i in range(num_of_stages)])
		
        return hg_list
    
    def _make_multi_stage_conv1x1(self, in_num_of_feat, out_num_of_feat, num_of_stages):

        conv1x1_list = nn.ModuleList([nn.Conv2d(in_num_of_feat, out_num_of_feat, 1, 1) for i in range(num_of_stages)])

        return conv1x1_list

    def forward(self, x):

        conf_pred_list = []
        orie_pred_list = []

        pool = self.basic_block(x)
        residual = pool
        out = self.res_block(pool)
        out += residual
        feat = self.res_pre_act(out)

        feat_to_next_stage = feat
        for i in range(self.num_of_stages):
			
            hg_out = self.hg_list[i](feat_to_next_stage)
            
            # For the first branch, predict the confidence maps			
            feat_1_b1 = self.conv1x1_1_list_b1[i](hg_out)
            feat_1_b1 = self.bn_b1[i](feat_1_b1)
            feat_1_b1 = self.relu(feat_1_b1)			
            conf_pred = self.conv_pred_list_b1[i](feat_1_b1)
            conf_pred_list.append(conf_pred)

            # For the second branch, predict the orientation maps
            feat_1_b2 = self.conv1x1_1_list_b2[i](hg_out)
            feat_1_b2 = self.bn_b2[i](feat_1_b2)
            feat_1_b2 = self.relu(feat_1_b2)
            orie_pred = self.conv_pred_list_b2[i](feat_1_b2)
            orie_pred_list.append(orie_pred)

            if i < self.num_of_stages - 1:
                # For the first branch, remamp feature
                feat_2_b1 = self.conv1x1_2_list_b1[i](feat_1_b1)
                feat_remap_b1 = self.conv_remap_list_b1[i](conf_pred)

                # For the second branch, remap feature
                feat_2_b2 = self.conv1x1_2_list_b2[i](feat_1_b2)
                feat_remap_b2 = self.conv_remap_list_b2[i](orie_pred)
                
                feat_to_next_stage = feat_to_next_stage + feat_2_b1 + feat_remap_b1 + feat_2_b2 + feat_remap_b2

        return conf_pred_list, orie_pred_list

# Hourglass network initialization with MSRA
def HG_MSRAInit(num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
    model = MSRAInit(HourglassNetwork(num_of_feat=num_of_feat, 
									  num_of_class=num_of_class, 
									  num_of_module=num_of_module, 
									  num_of_stages=num_of_stages))
    return model
 
# Hourglass network initialization with Gaussian
def HG_GaussianInit(num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
    model = GaussianInit(HourglassNetwork(num_of_feat=num_of_feat, 
										  num_of_class=num_of_class, 
										  num_of_module=num_of_module, 
										  num_of_stages=num_of_stages))
    return model

# GPN initialization with MSRA
def PPN_with_HG_MSRAInit(num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
    model = MSRAInit(PPN_with_HG(num_of_feat=num_of_feat, 
								 num_of_class=num_of_class, 
								 num_of_module=num_of_module, 
								 num_of_stages=num_of_stages))
    return model

# GPN initialization with Gaussian
def PPN_with_HG_GaussianInit(num_of_feat=256, num_of_class=17, num_of_module=1, num_of_stages=8):
    model = GaussianInit(PPN_with_HG(num_of_feat=num_of_feat, 
									 num_of_class=num_of_class, 
									 num_of_module=num_of_module, 
									 num_of_stages=num_of_stages))
    return model

if __name__ == '__main__':
    print('Pose Partition Networks with Hourglass achitecture as backbone')
