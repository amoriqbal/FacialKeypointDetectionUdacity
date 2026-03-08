## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernels=[3], stride=1):
        super(Block, self).__init__()
        self.conv_layers = []

        self.conv_layers += [
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
            ]
        for kernel in kernels:
            self.conv_layers += [
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False),
                nn.BatchNorm2d(out_channels)
            ]
        
        self.conv_layers = nn.ModuleList(self.conv_layers)
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride**2, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Sequential()   
 
    def forward(self, x):
        out = x
        for layer in self.conv_layers:
            out = layer(out)
        out = self.shortcut(x) + out
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 7, padding=3) 
        self.maxp1 = nn.MaxPool2d(4, 4) # // 4
        self.block1 = Block(32, 64, kernels=[3, 3])
        self.block2 = Block(64, 128, kernels=[3, 3])
        self.block3 = Block(128, 256, kernels=[3, 3])
        self.block4 = Block(256, 512, kernels=[3, 3], stride = 2) # //4
        self.block5 = Block(512, 512, kernels=[3, 3], stride = 2) # //4

        self.fc1 = nn.Linear(512*3*3, 2048)
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
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


# ## TODO: define the convolutional neural network architecture

# import torch.nn as nn
# class Block(nn.Module):
#     def __init__(self, in_channels, out_channels, kernels=[3], stride=1):
#         super(Block, self).__init__()
#         layers = [
#             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
#         ]
#         for kernel in kernels:
#             layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=stride, padding=kernel//2))
#         layers.append(nn.BatchNorm2d(out_channels))
#         layers.append(nn.ReLU())
#         self.sequence = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.sequence(x)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(1, 1)),
#             nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
#             nn.ReLU(),
#             Block(32,64, kernels=[3, 3]),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Block(64,128, kernels=[3, 3]),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Block(128,256, kernels=[3, 3]),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Block(256,512, kernels=[3, 3]),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             Block(512,512, kernels=[3, 3]),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#             nn.Flatten(start_dim=1, end_dim=-1),
#             nn.Linear(in_features=4608, out_features=2048, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=2048, out_features=512, bias=True),
#             nn.ReLU(),
#             nn.Linear(in_features=512, out_features=136, bias=True),
#         )

#     def forward(self, x):
#         return self.net(x)