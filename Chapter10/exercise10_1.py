import torch
import torch.nn as nn
import torch.nn.functional as F
CONV_FEATURES = 640
MID_FEATURES = 1024
SEC_FEATURES = 1024
OUTPUT_FEATURES = 10
# Passive setting -> flooring (in Pooling Layer)
# It's similar to LeNet.
# The general accuracy on cifar10 in LeNet is about 76%
# With 20 epoch, the accuracy keep increased (ended with 66%), so better to check about 30 epoch would be better.

class exercise_CNN(nn.Module):
    def __init__(self):
        # zero padding, one stirde.
        super(exercise_CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 3, stride = 1, padding=(1,1))
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = 3, stride = 1, padding = (1,1))
        self.conv3 = nn.Conv2d(in_channels = 20, out_channels = 40, kernel_size = 3, stride = 1, padding = (1,1))
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(CONV_FEATURES,MID_FEATURES)
        self.fc2 = nn.Linear(MID_FEATURES,SEC_FEATURES)
        self.fc3 = nn.Linear(SEC_FEATURES,OUTPUT_FEATURES)
        

    def forward(self,input):
        in_size = input.size(0)
        out = F.relu(self.pool(self.conv1(input)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        #Flatten
        out = out.view(in_size,-1)
        # I use the CE Loss so don't need to add softmax.
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
