# -*- coding: utf-8 -*-
import cv2
import numpy as np
import matplotlib.pyplot as plt


def  edge_prewitt(imgpath):
    # 读取图像
    img = cv2.imread(imgpath)

    img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    # 灰度化处理图像
    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = (img > 0).astype(int)

    # Prewitt算子
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
    x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)
    y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)
    # 转uint8
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    #考虑手动二值化并给他们编个号
    # ret, Prewitt = cv2.threshold(Prewitt, 0, 255, 0)
    #for循环遍历图像

    zhong = np.where(Prewitt > 0)

    savepath = imgpath.replace("Atrain_labels", "eage")
    cv2.imwrite(savepath, Prewitt)
    # # 用来正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图形
    # titles = [u'原始图像', u'Prewitt算子']
    # images = [img_RGB, Prewitt]
    # for i in range(2):
    #     plt.subplot(1, 2, i + 1), plt.imshow(images[i], 'gray')
    #     plt.title(titles[i])
    #     plt.xticks([]), plt.yticks([])
    # plt.show()


    # 显示图形
    # plt.subplot(121), plt.imshow(img_RGB), plt.title('原始图像'), plt.axis('off')  # 坐标轴关闭
    # plt.subplot(122), plt.imshow(Prewitt, cmap=plt.cm.gray), plt.title('Prewitt算子'), plt.axis('off')
    # plt.show()

    return Prewitt,zhong,a


#测试
# a,b = edge_prewitt("aaa.png")
# print(b)