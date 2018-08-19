import torch
import torch.nn as nn
import math
import time
import numpy as np

# Network initialization with Gaussian
class GaussianInit(nn.Module):

    def __init__(self, front_end_net):  
        super(GaussianInit, self).__init__()
        self.front_end_net = front_end_net
        self._initialize_weights()

    def forward(self, x):
        x = self.front_end_net(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

# Network initialization with Kaiming
class MSRAInit(nn.Module):

    def __init__(self, front_end_net):  
        super(MSRAInit, self).__init__()
        self.front_end_net = front_end_net
        self._initialize_weights()

    def forward(self, x):
        x = self.front_end_net(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal(m.weight, gain=1)
                m.bias.data.zero_()

if __name__ == '__main__':
    print("Network initialization")
        
