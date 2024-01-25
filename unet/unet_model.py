import torch.nn.functional as F

from unet.unet_parts import *
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
from BoundaryPreservingBlock.BoundaryPreservingBlock import *
from BoundaryPreservingBlock.chuli import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.downbpb2 = bpb(256)
        self.down3 = Down(256, 512)
        self.downbpb3 = bpb(512)
        self.down4 = Down(512, 1024)
        self.downbpb4 = bpb(1024)


        self.up1 = Up(1024, 512, bilinear)
        self.upbpb1 = bpb(512)
        self.up2 = Up(512, 256, bilinear)
        self.upbpb2 = bpb(256)
        self.up3 = Up(256, 128, bilinear)
        self.upbpb3 = bpb(128)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3, xx1 = self.downbpb2(x3)
        x4 = self.down3(x3)
        x4, xx2 = self.downbpb3(x4)
        x5 = self.down4(x4)
        x5, xx3= self.downbpb4(x5)
        x = self.up1(x5, x4)
        x, xx4 = self.upbpb1(x)
        x = self.up2(x, x3)
        x, xx5 = self.upbpb2(x)
        x = self.up3(x, x2)
        x, xx6 = self.upbpb3(x)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits,xx1,xx2,xx3,xx4,xx5,xx6

if __name__ == '__main__':
    net = UNet(n_channels=3, n_classes=1)
    # print(net)

    parametersList = []
    for param in net.named_parameters():
        if "bpb" not in param[0]:
            parametersList.append(param)
    # print(parametersList)
    bpbParametersList1 = []
    bpbParametersList2 = []
    bpbParametersList3 = []
    bpbParametersList4 = []
    bpbParametersList5 = []
    bpbParametersList6 = []
    for param in net.named_parameters():
        if "downbpb2" in param[0]:
            bpbParametersList1.append(param)
        if "downbpb3" in param[0]:
            bpbParametersList2.append(param)
        if "downbpb4" in param[0]:
            bpbParametersList3.append(param)
        if "upbpb1" in param[0]:
            bpbParametersList4.append(param)
        if "upbpb2" in param[0]:
            bpbParametersList5.append(param)
        if "upbpb3" in param[0]:
            bpbParametersList6.append(param)

    for name, param in parametersList:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList1:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList2:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList3:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList4:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList5:
        print(name)
    print("===============================================")
    for name, param in bpbParametersList6:
        print(name)

    img = torch.rand(1, 3, 512, 512)
    pred = net(img)
    print("=================!!!!!!!!!!!!!!!!!!!!!!!!++++++++++++++++++++++++")
    print(pred)
    print(pred.shape)