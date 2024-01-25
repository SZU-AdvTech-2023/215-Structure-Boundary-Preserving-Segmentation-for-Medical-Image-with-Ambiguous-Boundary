from shapely.geometry import Point
from shapely.geometry import Polygon
import numpy as np
import cv2

def getMask(imgpath,listarea):
    img = cv2.imread(imgpath)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp = np.ones(gray.shape, np.uint8) * 0  # 黑色背景

    # area_1 = Polygon([(0,0), (597, 233), (591, 245), (589, 256), (584, 255), (580, 249), (590, 233), (593, 225)])
    area_1 = Polygon(listarea)
    bbox = area_1.bounds
    arr_mask = []
    min_x, min_y, max_x, max_y = bbox[0], bbox[1], bbox[2], bbox[3]
    for i in range(int(min_x), int(max_x + 1)):
        for j in range(int(min_y), int(max_y + 1)):
            p_tmp = Point(i, j)
            if (p_tmp.within(area_1) == True):
                # print(i, j)
                pos_arr = [[i, j]]
                arr_mask.append(pos_arr)

    mmsk = np.array(arr_mask)
    cv2.drawContours(temp, mmsk, -1, (255, 255, 255), thickness=-1)

    cv2.waitKey(0)
    b = (temp > 0).astype(int)
    return b