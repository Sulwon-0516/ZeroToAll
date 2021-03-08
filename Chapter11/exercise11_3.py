# Simple DenseNET Model Generator. 
# The Only part it loses the equivalence of the NN is the last layer, so I also controlled it
# with getting the Input Size.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math 

COMPRESSION_RATE = 0.5

class unit_layer(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(unit_layer,self).__init__()
        # The 1x1 Conv output is always 4times of channel_out
        # channel_out : k, mid_depth = 4k
        self.BN1 = nn.BatchNorm2d(num_features = channel_in)
        self.Conv1x1 = nn.Conv2d(in_channels = channel_in, out_channels = 4 * channel_out, kernel_size = 1, stride = 1)
        self.BN2 = nn.BatchNorm2d(num_features = 4 * channel_out)
        self.Conv3x3 = nn.Conv2d(in_channels = 4 * channel_out, out_channels = channel_out, kernel_size = 3, stride = 1, padding = (1,1))

    def forward(self, input):
        out = F.relu(self.BN1(input))
        out = F.relu(self.BN2(self.Conv1x1(out)))
        out = self.Conv3x3(out)

        return out

class Dense_block(nn.Module):
    def __init__(self, channel_in, k, L):
        super(Dense_block,self).__init__()
        self.unit_list = []
        self.L = L
        for i in range(L):
            self.unit_list.append(unit_layer(math.floor(channel_in) + i*k, k))
        
    def forward(self, input):
        out = input
        for i in range(self.L):
            unit_out = self.unit_list[i](out)
            out = torch.cat((out,unit_out),axis = 1)
        
        return out

class Trans_layer(nn.Module):
    def __init__(self, channel_in):
        super(Trans_layer,self).__init__()
        channel_depth = math.floor(channel_in * COMPRESSION_RATE)
        self.BN = nn.BatchNorm2d(num_features = channel_in)
        self.Conv1x1 = nn.Conv2d(in_channels = channel_in, out_channels = channel_depth , kernel_size = 1, stride = 1)
        self.AvgPool = nn.AvgPool2d(kernel_size = 3, stride = 2, padding = (1,1))

    def forward(self, input):
        out = F.relu(self.BN(input))
        out = self.Conv1x1(out)
        out = self.AvgPool(out)

        return out

class DenseNet(nn.Module):
    def __init__(self, k, L, n_class):
        super(DenseNet,self).__init__()
        self.k = k
        self.L = L
        self.n_class = n_class

        self.Conv7x7_1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size = 7, stride = 1, padding = (3,3))
        self.MaxPool_1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = (1,1))
        
        self.Dense_1 = Dense_block(channel_in = 16, k = self.k[0], L = self.L[0])

        out_depth = self.L[0]*self.k[0]+16
        self.Trans_1 = Trans_layer(channel_in = out_depth)
        self.Dense_2 = Dense_block(channel_in = out_depth * COMPRESSION_RATE, k = self.k[1], L = self.L[1])

        out_depth = math.floor(out_depth * COMPRESSION_RATE)+self.L[1]*self.k[1]
        self.Trans_2 = Trans_layer(channel_in = out_depth)
        self.Dense_3 = Dense_block(channel_in = out_depth * COMPRESSION_RATE, k = self.k[2], L = self.L[2])

        out_depth = math.floor(out_depth * COMPRESSION_RATE)+self.L[2]*self.k[2]
        self.Trans_3 = Trans_layer(channel_in = out_depth)
        self.Dense_4 = Dense_block(channel_in = out_depth * COMPRESSION_RATE, k = self.k[3], L = self.L[3])

        out_depth = math.floor(out_depth * COMPRESSION_RATE)+self.L[3]*self.k[3]
        self.BN1 = nn.BatchNorm2d(num_features = out_depth)
        self.Conv1x1_5 = nn.Conv2d(in_channels = out_depth, out_channels = self.n_class, kernel_size = 1)

    def forward(self, input):
        
        out = self.Conv7x7_1(input)
        out = self.MaxPool_1(out)

        out = self.Dense_1(out)
        out = self.Trans_1(out)
        out = self.Dense_2(out)
        out = self.Trans_2(out)
        out = self.Dense_3(out)
        out = self.Trans_3(out)
        out = self.Dense_4(out)

        out = F.relu(self.BN1(out))
        out = self.Conv1x1_5(out)
        
        
        # I got the implemenetation method of "global pooling"
        # https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/18
        # it says that using torch.mean is faster than other Adaptive AvgPool or etc.
        # 0 is for batch and 1 is for the depth.
        out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)

        # or I can use kind of following things.
        #Avg_out = nn.AdaptiveAvgPool2d(1) 
        #out = Avg_out(out)

        out = torch.squeeze(out)

        return out


if __name__ == '__main__':
    # This is the general format of DensNet definition
    kargs = {"k" : [12,12,14,14], "L" : [6,12,24,16], "n_class" : 100 }
    model = DenseNet(kargs)
