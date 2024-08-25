# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 09:14:25 2023

@author: Administrator
"""
import cv2 as cv

import numpy as np
import random

def rand_drop_polygon(gray,drop_rate=0.01):
    """
    随机丢弃多边形
    """
    n, labels,_,_= cv.connectedComponentsWithStats(gray, connectivity=8)
    n_drop_building=int(np.ceil(n*drop_rate))
    #print(n_drop_building)
    neg=random.sample(range(n),n_drop_building)
    for i in neg:
        gray=np.where(labels==np.full_like(labels,i),np.full_like(gray,0),gray)
    return gray,n_drop_building # (h,w) np.U8

if __name__=='__main__':
    img=cv.imread(r'D:\Ajwl\data\Whan3_shuffle833\tiles_Images\all\label\3.tif',0)
    imgc=img.copy()
    imgn,n=rand_drop_polygon(imgc)
    cv.imshow('yuan',imgc)
    cv.imshow('label',imgn)
    cv.imshow('drop',img)
    cv.waitKey(0)
    cv.destroyAllWindows()

