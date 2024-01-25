import csv
from BoundaryKeyPointSelection import boundary_key_point_selection



def creatCsv(imgname):
    imgdirpath = "C:/Users/MLAI/Desktop/mydataset/Atrain_labels"
    csvpath = "C:/Users/MLAI/Desktop/mydataset/scv/"
    csvname = imgname.replace('jpg', 'csv')
    path = csvpath+csvname
    # path为输出路径和文件名，newline=''是为了不出现空行
    csvFile = open(path, "w+", newline='')
    # name为列名
    name = ['x', 'y']

    data = boundary_key_point_selection(imgdirpath+"/"+imgname)
    print("回来了吗")
    # print(data)

    try:
        writer = csv.writer(csvFile)
        writer.writerow(name)
        for i in data:
            writer.writerow(i)
    finally:
        csvFile.close()