

import Prewitt
import random
import getMask
import mathIOU
# import pandas as pd

#传入的参数是图像，或图像路径------暂定路径
def boundary_key_point_selection(imgpath):
    # print(imgpath)
    #a====>原图二值化后求iou的
    imgedge,zhong,a = Prewitt.edge_prewitt(imgpath)
    edgenum = len(zhong[0])

    IOUbest = 0

    bestAreList = []



    for i in range(401):

        #随机选取6个点
        count = 0
        dicnum = {}
        listare = []
        maxx = 0
        miny = 1000
        while count < 6:
            x = random.randrange(0, edgenum-1)
            if x not in dicnum:
                dicnum[x] = count
                a = zhong[0][x]
                b = zhong[1][x]
                tup = (b,a)
                listare.append(tup)
                if b > maxx:
                    maxx = b
                    miny = a
                elif b == maxx and a < miny:
                    miny = a

                count = count+1

        listare.remove((maxx,miny))
        sortnew = sorted(listare,key=lambda i:(i[1]-miny)/(i[0]-maxx-1))
        tub = (maxx, miny)
        sortnew.append(tub)
        #3.画出mask
        b = getMask.getMask(imgpath, sortnew)

        #计算IOU，找出最优的IOU
        ioua = mathIOU.iou(a, b)
        if ioua > IOUbest:
            IOUbest = ioua
            bestAreList= sortnew
    return bestAreList



