import torch
import torch.nn as nn
import torch.nn.functional as F

class boundaryPreservingBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        out_channels = 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=2, dilation=2)
        self.conv4 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=4, dilation=4)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=6, dilation=6)

        self.conv6 = nn.Conv2d(out_channels*5, out_channels, kernel_size=1, stride=1, padding=0, dilation=1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        xa = torch.cat((x1, x2, x3, x4, x5), 1)

        xa = self.conv6(xa)

        xa = self.Sigmoid(xa)

        return xa




