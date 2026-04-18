## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F

def BlockSequence(in_channels, out_channels, kernels=[3], stride=1):
    # sequence alternative to BlockCls
    sequence = []
    for i, kernel in enumerate(kernels):
        sequence.append(
            nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=kernel, stride=stride if i == 0 else 1, padding=kernel//2, bias=False)
        )
        sequence.append(nn.BatchNorm2d(out_channels))
        sequence.append(nn.ReLU(inplace=True))
    return nn.Sequential(*sequence)

Block = BlockSequence

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=2) # 224 -> 224
        self.maxp1 = nn.MaxPool2d(4, 4) # 224 -> 112
        self.block1 = Block(32, 64, kernels=[3, 3]) # 112 -> 112
        self.block2 = Block(64, 128, kernels=[3, 3]) # 56 -> 56
        self.block3 = Block(128, 256, kernels=[3, 3]) # 28 -> 28
        self.block4 = Block(256, 512, kernels=[3, 3]) # 14 -> 14
        self.block5 = Block(512, 512, kernels=[3, 3], stride=2) # 7 -> 7
        self.block6 = Block(512, 512, kernels=[3, 3], stride=2) # 7 -> 7
        self.avgp = nn.AdaptiveAvgPool2d((3, 3))
        self.fc1 = nn.Linear(512 * 3 * 3 , 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 136)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxp1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.avgp(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x