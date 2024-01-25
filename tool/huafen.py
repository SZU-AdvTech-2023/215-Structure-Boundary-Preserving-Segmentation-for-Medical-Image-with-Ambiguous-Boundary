import os,sys
import random
from PIL import Image

imgpath = "C:/Users/MLAI/Desktop/img"
maskpath = "C:/Users/MLAI/Desktop/mask"
imgdir = os.listdir(imgpath)

nums = 40

pred = random.sample(imgdir, nums)
print(pred)

for Afile in pred:
    predPath = "C:/Users/MLAI/Desktop/pre_img"
    im = Image.open(imgpath + "/" + Afile)
    im.save(predPath + "/" + Afile)
    os.remove(imgpath + "/" + Afile)


    pmaskPath = "C:/Users/MLAI/Desktop/pre_mask"
    imask = Image.open(maskpath + "/" + Afile)
    imask.save(pmaskPath + "/" + Afile)
    os.remove(maskpath + "/" + Afile)

