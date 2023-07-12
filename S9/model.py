import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 32 | 32

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 32 | 30
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # 30 | 28
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 28 | 28
        
        '''
        self.pool1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride = 2, padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # 30 | 28 # ip = 28 | op = 14 
        '''
        # (No MaxPooling, but 3 3x3 layers with stride of 2 instead) (Attempt at using Dilation along with strided convolution)
        self.strided_pooling = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 2), stride = 2, padding=2, dilation = 4, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # ip = 28 | op = 14 

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=100, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(100)
        ) # 14 | 12

        # Dilated Convolution 1
        self.dialted_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=64, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64)
        ) # 12 | 10

        # Dilated Convolution 2
        self.dialted_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), padding=1, dilation = 2, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32)
        ) # 10 | 8

        # Depthwise Separable Convolution
        self.depthwise_separable_conv = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = (3, 3), padding = 0, groups = 32, bias = False),
            nn.Conv2d(in_channels = 32, out_channels = 20, kernel_size = (1, 1), padding = 0, bias = False),
            nn.ReLU(),
            nn.BatchNorm2d(20)
        ) # 8 | 6

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16)
        ) # 6 | 4

        # GAP layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=4)
        ) # ip = 10 | op = 1 

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # ip = 1 | op = 1 | RF = 28

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.strided_pooling(x)
        x = self.conv5(x)
        x = self.dialted_conv_1(x)
        x = self.dialted_conv_2(x)
        x = self.depthwise_separable_conv(x)
        x = self.conv6(x)
        x = self.gap(x)
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
