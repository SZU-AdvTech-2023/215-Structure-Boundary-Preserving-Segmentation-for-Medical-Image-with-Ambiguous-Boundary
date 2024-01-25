import numpy as np
import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import random


class ISBI_Loader(Dataset):
    def __init__(self, data_path):
        # 初始化函数，读取所有data_path下的图片
        self.data_path = data_path
        self.imgs_path = glob.glob(os.path.join(data_path, 'Atrain_imgs/*.jpg'))
        print('===========数据加载完毕========')

    def augment(self, image, flipCode):
        # 使用cv2.flip进行数据增强，filpCode为1水平翻转，0垂直翻转，-1水平+垂直翻转
        flip = cv2.flip(image, flipCode)
        return flip

    def __getitem__(self, index):
        # 根据index读取图片
        image_path = self.imgs_path[index]
        # 根据image_path生成label_path
        label_path = image_path.replace('Atrain_imgs', 'Atrain_labels')
        label1_path = image_path.replace('Atrain_imgs', '2conv3')
        label2_path = image_path.replace('Atrain_imgs', '2conv4')
        label3_path = image_path.replace('Atrain_imgs', '2conv5')
        label4_path = image_path.replace('Atrain_imgs', '2conv2')

        # 读取训练图片和标签图片
        image = cv2.imread(image_path)
        label = cv2.imread(label_path)
        label1 = cv2.imread(label1_path)
        label2 = cv2.imread(label2_path)
        label3 = cv2.imread(label3_path)
        label4 = cv2.imread(label4_path)

        image = cv2.resize(image, (765, 572), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (765, 572), interpolation=cv2.INTER_NEAREST)
        image = np.transpose(image, (2,0,1))

        # 将数据转为单通道的图片
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label1 = cv2.cvtColor(label1, cv2.COLOR_BGR2GRAY)
        label2 = cv2.cvtColor(label2, cv2.COLOR_BGR2GRAY)
        label3 = cv2.cvtColor(label3, cv2.COLOR_BGR2GRAY)
        label4 = cv2.cvtColor(label4, cv2.COLOR_BGR2GRAY)


        image = image / 255.0

        # 处理标签，将像素值为255的改为1
        if label.max() > 1:
            label = label / 255
        if label1.max() > 1:
            label1 = label1 / 255
        if label2.max() > 1:
            label2 = label2 / 255
        if label3.max() > 1:
            label3 = label3 / 255
        if label4.max() > 1:
            label4 = label4 / 255


        label = np.expand_dims(label, axis=0)
        label1 = np.expand_dims(label1, axis=0)
        label2 = np.expand_dims(label2, axis=0)
        label3 = np.expand_dims(label3, axis=0)
        label4 = np.expand_dims(label4, axis=0)

        return image, label, label1,label2,label3,label4

    def __len__(self):
        # 返回训练集大小
        print(len(self.imgs_path))
        return len(self.imgs_path)


if __name__ == "__main__":
    isbi_dataset = ISBI_Loader("C:/Users/MLAI/Desktop/mydataset")
    print("数据个数：", len(isbi_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=2,
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)