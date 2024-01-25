import os
import cv2

from PIL import Image

# 打开文件
path = "C:/Users/MLAI/Desktop/aaa"
dirs = os.listdir(path)
file_path = "C:/Users/MLAI/Desktop/shishi"

Apath = "C:/Users/MLAI/Desktop/PH2Dataset/PH2Dataset/PH2 Dataset images"
Adirs = os.listdir(Apath)

# # 输出所有文件和文件夹
# for file in dirs:
#     newFileName = file[0:file.find(".")] + ".jpg"
#     newFileName = newFileName.replace('_lesion', '')
#     im = Image.open(path + "/" + file)
#     im.save(file_path + "/" + newFileName)
#     print(file)


# 输出所有文件和文件夹-----这是一级目录，到图像的最外层目录
for Afile in Adirs:
    imgpath = Apath+"/"+Afile+"/"+Afile+"_Dermoscopic_Image"
    bdirs = os.listdir(imgpath)
    newimgpath = "C:/Users/MLAI/Desktop/img"
    for imgFile in bdirs:
        newImgNeme = imgFile[0:imgFile.find(".")] + ".jpg"
        im = Image.open(imgpath + "/" + imgFile)
        im.save(newimgpath + "/" + newImgNeme)

    imgpath = Apath+"/"+Afile+"/"+Afile+"_lesion"
    bdirs = os.listdir(imgpath)
    newimgpath = "C:/Users/MLAI/Desktop/mask"
    for imgFile in bdirs:
        newImgNeme = imgFile[0:imgFile.find(".")] + ".jpg"
        newImgNeme = newImgNeme.replace('_lesion', '')
        im = Image.open(imgpath + "/" + imgFile)
        im.save(newimgpath + "/" + newImgNeme)