# Implementing Full Inception v4 basic model.
import torch
import torch.nn as nn
import torch.nn.functional as F 

K = 192
L = 224
M = 256
N = 384


class Inception_stem(nn.Module):
    def __init__(self):
        super(Inception_stem,self).__init__()
        self.Conv3_1 = nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2)
        self.Conv3_2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1)
        self.Conv3_3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding=(1,1))

        self.MaxPool_A_1 = nn.MaxPool2d(kernel_size = 3, stride = 2)
        self.Conv3_B_1 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 2)

        self.Conv1_A_2_1 = nn.Conv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1)
        self.Conv3_A_2_2 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1)

        self.Conv1_B_2_1 = nn.Conv2d(in_channels = 160, out_channels = 64, kernel_size = 1, stride = 1) 
        self.Conv1_B_2_2 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (7,1), stride = 1, padding=(3,0)) 
        self.Conv1_B_2_3 = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = (1,7), stride = 1, padding=(0,3)) 
        self.Conv1_B_2_4 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1)

        self.Conv3_A_3 = nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 2)  
        self.MaxPool_B_3 = nn.MaxPool2d(kernel_size = 3, stride = 2) 

    def forward(self,input):
        out = F.relu(self.Conv3_1(input))
        out = F.relu(self.Conv3_2(out))
        out = self.Conv3_3(out)

        outA = F.relu(self.MaxPool_A_1(out))
        outB = F.relu(self.Conv3_B_1(F.relu(out)))
        out = torch.cat((outA,outB),axis =1)
        #print(out.size())
        outA = F.relu(self.Conv1_A_2_1(out))
        outA = self.Conv3_A_2_2(outA)
        outB = F.relu(self.Conv1_B_2_1(out))
        outB = F.relu(self.Conv1_B_2_2(outB))
        outB = F.relu(self.Conv1_B_2_3(outB))
        outB = self.Conv1_B_2_4(outB)
        out = torch.cat((outA,outB),axis =1)
        #print(out.size())
        outA = self.Conv3_A_3(F.relu(out))
        outB = self.MaxPool_B_3(out)
        out = torch.cat((outA,outB),axis =1)
        #print(out.size())
        return out

class Inception_module_A(nn.Module):
    def __init__(self,in_channels):
        super(Inception_module_A,self).__init__()
        # the avg pooling layers are stride 1, pad 1
        self.AvgPool_A_1 = nn.AvgPool2d(kernel_size = 3, stride =1, padding = (1,1))
        self.Conv_A_2 = nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 1, stride = 1)

        self.Conv_B_1 = nn.Conv2d(in_channels = in_channels, out_channels = 96, kernel_size = 1, stride = 1)

        self.Conv_C_1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1)
        self.Conv_C_2 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = (1,1))

        self.Conv_D_1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 1, stride = 1)
        self.Conv_D_2 = nn.Conv2d(in_channels = 64, out_channels = 96, kernel_size = 3, stride = 1, padding = (1,1))
        self.Conv_D_3 = nn.Conv2d(in_channels = 96, out_channels = 96, kernel_size = 3, stride = 1, padding = (1,1))

    def forward(self, input):
        outA = self.AvgPool_A_1(input)
        outA = self.Conv_A_2(outA)

        outB = self.Conv_B_1(input)

        outC = F.relu(self.Conv_C_1(input))
        outC = self.Conv_C_2(outC)

        outD = F.relu(self.Conv_D_1(input))
        outD = F.relu(self.Conv_D_2(outD))
        outD = self.Conv_D_3(outD)

        out = torch.cat((outA, outB, outC, outD),axis=1)
        return out

class Inception_module_B(nn.Module):
    def __init__(self,in_channels):
        super(Inception_module_B,self).__init__()
        # the avg pooling layers are stride 1, pad 1
        self.AvgPool_A_1 = nn.AvgPool2d(kernel_size = 3, stride =1, padding = (1,1))
        self.Conv_A_2 = nn.Conv2d(in_channels = in_channels, out_channels = 128, kernel_size = 1, stride = 1)

        self.Conv_B_1 = nn.Conv2d(in_channels = in_channels, out_channels = 384, kernel_size = 1, stride = 1)

        self.Conv_C_1 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 1)
        self.Conv_C_2 = nn.Conv2d(in_channels = 192, out_channels = 224, kernel_size = (7,1), stride = 1, padding = (3,0))
        self.Conv_C_3 = nn.Conv2d(in_channels = 224, out_channels = 256, kernel_size = (1,7), stride = 1, padding = (0,3))

        self.Conv_D_1 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 1, stride = 1)
        self.Conv_D_2 = nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = (1,7), stride = 1, padding = (3,0))
        self.Conv_D_3 = nn.Conv2d(in_channels = 192, out_channels = 224, kernel_size = (7,1), stride = 1, padding = (0,3))
        self.Conv_D_4 = nn.Conv2d(in_channels = 224, out_channels = 224, kernel_size = (1,7), stride = 1, padding = (3,0))
        self.Conv_D_5 = nn.Conv2d(in_channels = 224, out_channels = 256, kernel_size = (7,1), stride = 1, padding = (0,3))

    def forward(self, input):
        outA = self.AvgPool_A_1(input)
        outA = self.Conv_A_2(outA)

        outB = self.Conv_B_1(input)

        outC = F.relu(self.Conv_C_1(input))
        outC = F.relu(self.Conv_C_2(outC))
        outC = self.Conv_C_3(outC)

        outD = F.relu(self.Conv_D_1(input))
        outD = F.relu(self.Conv_D_2(outD))
        outD = F.relu(self.Conv_D_3(outD))
        outD = F.relu(self.Conv_D_4(outD))
        outD = self.Conv_D_5(outD)

        out = torch.cat((outA, outB, outC, outD),axis=1)
        return out

class Inception_module_C(nn.Module):
    def __init__(self,in_channels):
        super(Inception_module_C, self).__init__()
        self.AvgPool_A_1 = nn.AvgPool2d(kernel_size = 3, stride =1, padding = (1,1))
        self.Conv_A_2 = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 1, stride = 1)

        self.Conv_B_1 = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 1, stride = 1)

        self.Conv_C_1 = nn.Conv2d(in_channels = in_channels, out_channels = 384, kernel_size = 1, stride = 1)
        self.Conv_C_2_A = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = (3,1), stride = 1, padding = (1,0))
        self.Conv_C_2_B = nn.Conv2d(in_channels = 384, out_channels = 256, kernel_size = (1,3), stride = 1, padding = (0,1))

        self.Conv_D_1 = nn.Conv2d(in_channels = in_channels, out_channels = 384, kernel_size = 1, stride = 1)
        self.Conv_D_2 = nn.Conv2d(in_channels = 384, out_channels = 448, kernel_size = (1,3), stride = 1, padding = (0,1))
        self.Conv_D_3 = nn.Conv2d(in_channels = 448, out_channels = 512, kernel_size = (3,1), stride = 1, padding = (1,0))
        self.Conv_D_4_A = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (3,1), stride = 1, padding = (1,0))
        self.Conv_D_4_B = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = (1,3), stride = 1, padding = (0,1))

    def forward(self,input):
        outA = self.AvgPool_A_1(input)
        outA = self.Conv_A_2(outA)

        outB = self.Conv_B_1(input)

        outC = F.relu(self.Conv_C_1(input))
        outC_A = self.Conv_C_2_A(outC)
        outC_B = self.Conv_C_2_B(outC)
        outC = torch.cat((outC_A,outC_B),axis=1)

        outD = F.relu(self.Conv_D_1(input))
        outD = F.relu(self.Conv_D_2(outD))
        outD = F.relu(self.Conv_D_3(outD))
        outD_A = self.Conv_D_4_A(outD)
        outD_B = self.Conv_D_4_B(outD)
        outD = torch.cat((outD_A,outD_B),axis=1)

        out = torch.cat((outA, outB, outC, outD),axis=1)
        return out

class Reduction_A(nn.Module):
    def __init__(self,in_channels):
        super(Reduction_A, self).__init__()
        self.MaxPool_A = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.Conv_B = nn.Conv2d(in_channels = in_channels, out_channels = N, kernel_size = 3, stride = 2)

        self.Conv_C_1 = nn.Conv2d(in_channels = in_channels, out_channels = K, kernel_size = 1, stride = 1)
        self.Conv_C_2 = nn.Conv2d(in_channels = K, out_channels = L, kernel_size = 3, stride = 1, padding = (1,1))
        self.Conv_C_3 = nn.Conv2d(in_channels = L, out_channels = M, kernel_size = 3, stride = 2)

    def forward(self, input):
        outA = self.MaxPool_A(input)

        outB = self.Conv_B(input)

        outC = F.relu(self.Conv_C_1(input))
        outC = F.relu(self.Conv_C_2(outC))
        outC = self.Conv_C_3(outC)

        out = torch.cat((outA, outB, outC),axis=1)
        return out

class Reduction_B(nn.Module):
    def __init__(self,in_channels):
        super(Reduction_B, self).__init__()
        self.MaxPool_A = nn.MaxPool2d(kernel_size = 3, stride = 2)
        
        self.Conv_B_1 = nn.Conv2d(in_channels = in_channels, out_channels = 192, kernel_size = 1, stride = 1)
        self.Conv_B_2 = nn.Conv2d(in_channels = 192, out_channels = 192, kernel_size = 3, stride = 2)

        self.Conv_C_1 = nn.Conv2d(in_channels = in_channels, out_channels = 256, kernel_size = 1, stride = 1)
        self.Conv_C_2 = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = (1,7), stride = 1, padding = (0,3))
        self.Conv_C_3 = nn.Conv2d(in_channels = 256, out_channels = 320, kernel_size = (7,1), stride = 1, padding = (3,0))
        self.Conv_C_4 = nn.Conv2d(in_channels = 320, out_channels = 320, kernel_size = 3, stride= 2)

    def forward(self, input):
        outA = self.MaxPool_A(input)

        outB = F.relu(self.Conv_B_1(input))
        outB = self.Conv_B_2(outB)

        outC = F.relu(self.Conv_C_1(input))
        outC = F.relu(self.Conv_C_2(outC))
        outC = F.relu(self.Conv_C_3(outC))
        outC = self.Conv_C_4(outC)

        out = torch.cat((outA, outB, outC),axis=1)
        return out

class Inception_v4(nn.Module):
    def __init__(self):
        super(Inception_v4,self).__init__()
        
        self.stem = Inception_stem()

        self.Inc_A_1 = Inception_module_A(384)
        self.Inc_A_2 = Inception_module_A(384)
        self.Inc_A_3 = Inception_module_A(384)
        self.Inc_A_4 = Inception_module_A(384)

        self.Red_A = Reduction_A(384)

        self.Inc_B_1 = Inception_module_B(1024)
        self.Inc_B_2 = Inception_module_B(1024)
        self.Inc_B_3 = Inception_module_B(1024)
        self.Inc_B_4 = Inception_module_B(1024)

        self.Red_B = Reduction_B(1024)

        self.Inc_C_1 = Inception_module_C(1536)
        self.Inc_C_2 = Inception_module_C(1536)
        self.Inc_C_3 = Inception_module_C(1536)
        self.Inc_C_4 = Inception_module_C(1536)

        self.AvgPool = nn.AvgPool2d(kernel_size = 8)
        self.Dropout = nn.Dropout(p=0.2)
        self.Linear = nn.Linear(1536,1000)
    
    def forward(self,input):
        out = self.stem(input)

        out = self.Inc_A_1(out)
        out = self.Inc_A_2(out)
        out = self.Inc_A_3(out)
        out = self.Inc_A_4(out)

        out = self.Red_A(out)

        out = self.Inc_B_1(out)
        out = self.Inc_B_2(out)
        out = self.Inc_B_3(out)
        out = self.Inc_B_4(out)

        out = self.Red_B(out)

        out = self.Inc_C_1(out)
        out = self.Inc_C_2(out)
        out = self.Inc_C_3(out)
        out = self.Inc_C_4(out)

        out = self.AvgPool(out)
        out = out.squeeze()
        out = self.Dropout(out)
        out = self.Linear(out)

        return out

