import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image



scvpath = "C:/Users/MLAI/Desktop/mydataset/scv"
cvdir = os.listdir(scvpath)
pointmask = "C:/Users/MLAI/Desktop/mydataset/keypointimg"



#读入一涨mask看大小---765*572
file_path = 'C:/Users/MLAI/Desktop/mydataset/Atrain_labels/IMD002.jpg'
img = Image.open(file_path)
imgSize = img.size  # 大小/尺寸
w = img.width  # 图片的宽
h = img.height  # 图片的高
print(w,h)
i = 1

for filename in cvdir:
    # 读取scv文件
    #数据形式[[456. 293.][372. 305.][305. 297.][238. 284.][468. 169.][521. 259.]]
    keypointdata = np.loadtxt(open(scvpath+"/"+filename,"rb"), delimiter=",", skiprows=1, usecols=[0,1])
    print(keypointdata[0])
    #生成一样大小的画幅，生成时是黑色的
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # print(img.shape)
    #修改颜色-->在关键点半径3个点
    for wa in range(0, w):
        for ha in range(0, h):
            for point in keypointdata:
                # print(point[0],"===================================",point[1])
                a = 3
                if wa >= point[0]-a and wa <= point[0]+a and ha >= point[1]-a and ha <= point[1]+a:
                    img[ha, wa] = (255, 255, 255)
                    # print("true")
    bb = filename.replace('csv', 'jpg')
    pointmaskpath = pointmask+"/"+bb
    print(pointmaskpath)
    cv2.imwrite(pointmaskpath, img)
    # print(keypointdata)
    print(i)
    i = i+1