from createcsv import creatCsv
import os

imgdirpath = "C:/Users/MLAI/Desktop/mydataset/Atrain_imgs"
csvpath = "C:/Users/MLAI/Desktop/mydataset/scv/"
files = os.listdir(imgdirpath)

for file in files:
    creatCsv(file)
