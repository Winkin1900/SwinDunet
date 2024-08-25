# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 19:32:11 2024

输入的是 change[b,2,h,w] tensor float32
任务：去掉edge
输出：del_edge_change[b,2,h,w] tensor float32

@author: jwl
"""

import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt



def removed_edge(change,drop_rate=None,threshold=200):#[b,2,h,w]tf32
    # 
    del_edge_change=[]
    change=tensor2array(change)
    for rn in change:
        removed,new = rn[...,0],rn[...,1]#[h,w]nu8
        # 筛选方式是阈值
        if drop_rate is None:
            removed = del_edge_by_area(removed,threshold)
            new     = del_edge_by_area(new, threshold)
        # 筛选方式是个数
        else:
            removed = del_edge_by_n(removed,drop_rate)
            new     = del_edge_by_n(new, drop_rate)
        #
        rn_del=np.stack((removed,new),axis=0)#[2,h,w]
        del_edge_change.append(rn_del)
# =============================================================================
#         plt.imshow(removed)
#         plt.show()
#         plt.imshow(new)
#         plt.show()
# =============================================================================
    del_edge_change=np.array(del_edge_change)#[b,2,h,w]
    del_edge_change=torch.from_numpy(del_edge_change.astype(np.float32))/255
    return del_edge_change
    

def del_edge_by_n(image,drop_rate):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    n = int(np.ceil(num_labels*drop_rate))
    areas=stats[1:,4] # 去掉背景
    inlist = np.argsort(areas) 
    areas_sorted =areas[inlist]
    #print(areas_sorted[n:])
    out = np.where(np.isin(labels, (inlist[n:]+1)),255,0).astype(np.uint8)
    return out

def del_edge_by_area(image,threshold):
    num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    
    filtered_image = np.zeros_like(labels)
    for label_id, stat in enumerate(stats):
        if label_id == 0:
            continue
        #
        left, top, width, height,area= stat
        if width < height:
            width, height = height, width
        aspect_ratio = width / height if height != 0 else 0
        # 设置阈值
        if  area > threshold:#aspect_ratio <= 10 and
           filtered_image[labels == label_id] = 255
    filtered_image = np.uint8(filtered_image)
    return filtered_image

def tensor2array(t):
    #(b,c,h,w)-->(b,h,w,c)U8
    assert len(t.shape) == 4,'shape not match !'
    t=t.permute(0,2,3,1)
    t=np.array(t*255,dtype=np.uint8)
    return t 


