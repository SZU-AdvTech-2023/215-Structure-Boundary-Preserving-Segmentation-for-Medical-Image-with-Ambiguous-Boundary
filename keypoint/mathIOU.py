import numpy as np
import cv2


def iou(apath, bpath, epsilon=1e-5):

    a = apath
    b = bpath

    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou
