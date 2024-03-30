import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class PreConvNet(nn.Module):
    def __init__(self):
        super(PreConvNet,self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(5, 4, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(7, stride=1, padding=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x