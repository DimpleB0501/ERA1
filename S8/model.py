from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm2d

dropout_value = 0

class Net(nn.Module):
    def __init__(self, normalization_type = "Batch", group_num = 0):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = self.convolution_layer (1, 8, 3, normalization_type, group_num)
        # CONVOLUTION BLOCK 
        self.convblock2 = self.convolution_layer (8, 24,  3, normalization_type, group_num)

        # TRANSITION BLOCK 
        self.transblock1 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=18, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
        ) 
        self.pool1 = nn.MaxPool2d(2, 2) 

        # CONVOLUTION BLOCK 
        self.convblock3 = self.convolution_layer (18, 16,  3, normalization_type, group_num)
        

        # CONVOLUTION BLOCK 
        self.convblock4 = self.convolution_layer (16, 16,  3, normalization_type, group_num)
      

        # CONVOLUTION BLOCK
        self.convblock5 = self.convolution_layer (16, 16,  3, normalization_type, group_num)
        
        # GAP layer
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) 
        self.transblock2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) 
        
    def batchnormalization(self, output_channel_size, normalization_type = "Batch", group_num=0):
      if normalization_type == "Batch":
        return nn.BatchNorm2d(output_channel_size)
      elif normalization_type == "Layer": # https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html
        return nn.GroupNorm(1, output_channel_size)
      elif normalization_type == "Group":
        return nn.GroupNorm(group_num, output_channel_size)

    def convolution_layer (self, input, output, kernel_val=3, normalization_type="Batch", group_num=0, dropout_val=0):
        layer = nn.Sequential(
                  nn.Conv2d(in_channels=input, out_channels=output, kernel_size=(kernel_val, kernel_val), padding=0, bias=False),
                  nn.ReLU(),
                  self.batchnormalization(output, normalization_type, group_num),
                  nn.Dropout(dropout_value)
                )  
        return layer


    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.transblock1(x)
        x = self.pool1(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.gap(x)
        x = self.transblock2(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)
