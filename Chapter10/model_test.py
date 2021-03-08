import torch
import torch.nn as nn
import torch.nn.functional as F
FINAL_FEATURES = 720
OUTPUT_FEATURES = 10

class simple_CNN(nn.Module):
    def __init__(self):
        # zero padding, one stirde.
        super(simple_CNN,self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 10, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv2d(in_channels = 10, out_channels = 20, kernel_size = (3,3), stride = 1)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(FINAL_FEATURES,OUTPUT_FEATURES)

    def forward(self,input):
        in_size = input.size(0)
        out = F.relu(self.pool(self.conv1(input)))
        out = F.relu(self.pool(self.conv2(out)))
        out = F.relu(self.pool(self.conv3(out)))
        #Flatten
        out = out.view(in_size,-1)
        # I use the CE Loss so don't need to add softmax.
        out = self.fc(out)
        return out
