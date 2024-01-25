import torch
import torch.nn as nn
from PIL import Image
import cv2
import numpy as np
import os

class Neta(nn.Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.down = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.down(x)
        # x = self.down(x)
        # x = self.down(x)
        return x

if __name__ == '__main__':
    net = Neta(1, 1)
    # file_path = 'C:/Users/MLAI/Desktop/mydataset/Atrain_labels/IMD002.jpg'
    # img = Image.open(file_path)
    file_path = 'C:/Users/MLAI/Desktop/mydataset/keypointimg'
    filedir = os.listdir(file_path)
    pathimg3conv = "C:/Users/MLAI/Desktop/mydataset/2conv2"
    i = 1

    for filename in filedir:
        imgpatha = file_path+"/"+filename
        # img = Image.open(imgpatha)
        img = cv2.imread(imgpatha)
        # print("==========", img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # print("==========",img.shape)


        img = torch.from_numpy(img)
        img = torch.tensor(img, dtype=torch.float32)
        # img = torch.unsqueeze(img, dim=0)
        img = torch.unsqueeze(img, dim=0)
        # print(img.shape)
        pred = net(img)
        # print(pred.shape)
        pred = np.array(pred.data.cpu()[0])
        # print("np.array(pred.data==========", pred.shape)

        pred = np.array(pred)
        pred[pred >= 0.5] = 255
        pred[pred < 0.5] = 0
        # print(pred)
        # print(pred.shape)
        # cv2.imwrite("shishiaaa.jpg", pred)
        # pred = cv2.resize(pred, (765, 572), interpolation=cv2.INTER_NEAREST)
        # cv2.imwrite("shishiaaabbb.jpg", pred)

        cv2.imwrite(pathimg3conv+"/"+filename, pred)
        print(i)
        i = i+1

