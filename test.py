# -*- coding: utf-8 -*-
"""
-------------------------------------------------
Project Name: unet
File Name: test.py
Author: chenming
Create Date: 2022/2/7
Description：
-------------------------------------------------
"""

import numpy as np
import os
import cv2
from unet.unet_model import UNet
import torch,gc
from tqdm import tqdm


def cal_miou(test_dir="/data/jiayandai/mydataset/pre_img",
             pred_dir="/data/jiayandai/mydataset/2BPBpre3", gt_dir="/root/dataset/Atest_labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "nidus"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = UNet(n_channels=3, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model_.pth', map_location=device)) # todo
        # 测试模式
        # net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            img = cv2.imread(image_path)
            origin_shape = img.shape
            # print(origin_shape)
            # 转为灰度图
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img= cv2.resize(img, (765, 572), interpolation=cv2.INTER_NEAREST)
            # img = cv2.resize(img, (512, 512))
            
            # 转为batch为1，通道为1，大小为512*512的数组
            # img = img.reshape(1, 1, img.shape[0], img.shape[1])
            img =np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)


            img = img / 255.0
            
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            # 提取结果
            pred = pred[0]
            print(pred)
            print(pred.shape)
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(pred_dir, image_id + ".jpg"), pred)

        print("Get predict result done.")


if __name__ == '__main__':
    cal_miou()