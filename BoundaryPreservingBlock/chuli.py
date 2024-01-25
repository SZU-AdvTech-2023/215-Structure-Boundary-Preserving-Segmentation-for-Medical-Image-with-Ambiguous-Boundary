import torch.nn as nn
from BoundaryPreservingBlock.BoundaryPreservingBlock import boundaryPreservingBlock


#完成 vi =fi ⊕ (fi ⊗ ˆMi) 并返回
class bpb(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.bpb1 = boundaryPreservingBlock(in_channels)

    def forward(self, x):
        x1 = self.bpb1(x)

        x2 = x1*x

        x3 = x2+x

        return x3, x1


# model = bpb()
#
# x3, x1 = model(x)
#
# loss = loss1(x3) + loss2(x1)