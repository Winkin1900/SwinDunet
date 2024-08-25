# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:18:50 2023
作用于数据集的变形长函数
像素坐标系的x是竖着的h,y是横着的w
@author: l
"""
    
import numpy as np
import cv2 as cv
import os,sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"# OMP: Error #15

class SpatialTransformer(nn.Module):
    def __init__(self,size,mode="nearest"):
        """
        size:(320,480).网格大小。
        mode：F.grid-sample的采样方法，有'bilinear','nearest'.
        return:F.grid_sample：输出形变后的张量(1,3,h,w)
        """
        super().__init__()
        self.size=size
        vectors=[torch.arange(0,s)for s in size]
        x,y=torch.meshgrid(vectors,indexing='ij')
        grid=torch.stack((x,y))#(2,h,w)
        grid= torch.unsqueeze(grid, 0)#(1,2,h,w)
        grid= grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)#不参与参数更新,但是被加入到模型参数
        self.mode = mode
        
    def forward(self,src,df):
        """
        src:tensor(1,3,h,w),float32,(0->1)
        df:tensor(1,2,h,w),float32,(0-255)
        grid指定了采样像素的位置，这里的位置是由input的空间维度归一化后的结果。
        所以范围在[-1, 1]区间内。比如x=-1, y=-1表示输入的最左上角的位置，x=1,
         y=1表示输入的最右下角位置。
         return (b,3,h,w)
        """
        #src=array2tensor(src) 
        #df=array2tensor(df*255)#抵消一下
        self.grid=self.grid.expand(df.shape)
        new_grid=self.grid+df#(1,2,h,w)
        for i in range(len(self.size)):
            new_grid[:,i,...]=2*(new_grid[:,i,...]/(self.size[i]-1)-0.5)#
            #为实现F.grid_sample,实现空间维度归一化。
        new_grid=new_grid.permute(0,2,3,1)
        new_grid=new_grid[...,[1,0]]
        return F.grid_sample(src,new_grid,mode=self.mode,padding_mode='zeros',
                             align_corners=False)

def array2tensor(img):
    img=torch.from_numpy(img.astype(np.float32))
    t=img.unsqueeze(0).permute(0,3,1,2)
    return t

def tensor2array(t):
    #(1,c,h,w)-->(h,w,c)U8
    assert len(t.shape)==4,'shape not match !'
    if len(t.shape)==4:
        t=t.squeeze(0).permute(1,2,0)
        t=np.array(t*255,dtype=np.uint8)
    return t   
        
if __name__=='__main__':   
    print('jwl')