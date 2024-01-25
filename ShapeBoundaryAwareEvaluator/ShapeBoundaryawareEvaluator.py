import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super.__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __int__(self,in_channels, out_channels):
        self.maxpool_conv = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class BSE(nn.Module):
    def __init__(self,n_channels):
        super.__init__()
        self.n_channels = n_channels
        # self.n_classes = n_classes

        self.down1 = Down(n_channels,64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = DoubleConv(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(512,256)

    def forward(self, x):
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.avgpool(x)
        x = self.linear(x)