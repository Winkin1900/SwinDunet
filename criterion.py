# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:05:19 2023
损失函数
s_y的通道数做了调整，数据集那边还没有对接。
F.interplate函数输入必须是四维浮点数,align_corners=False方便对齐
@author: l
"""
import torch,sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
sys.path.append(r'.\dataset\utils')
from apply_deformation import SpatialTransformer
import matplotlib.pyplot as plt

class criterion(nn.Module):
    def __init__(self,w_c,wc_hat,alpha,sigma):
        """
        Parameters
        ----------
        w_c : tensor.float32 [3]
            配准损失的类别系数
        wc_hat : tensor.float32 [3]
            分割损失的 类别系数
        alpha : tensor.float32 [3]
            三层损失的系数
        """
        super().__init__()
        self.sigma=sigma
        self.w_c=torch.tensor([10,10,10]).cuda()
        self.wc_hat=torch.tensor([0.05,0.1,1,10]).cuda()
        self.alpha=torch.tensor([0.5,0.35,0.15]).cuda()
        self.gradloss = Grad('l2',None).loss
    
    def forward(self,y_rgted,s_Y1,s_y,s_Y2,flow):
        """
        Parameters
        ----------
        s_x : tensor.float32 list(3) [b,3,h,w] [b,3,h/2,w/2] [b,3,h/4,w/4]
            分割的三级结果，
        s_y : tensor.float32 [b,4,h,w]  (0,1)
            分割的标签，未分级。
        a_x : tensor.float32 list(3) [b,2,h,w] [b,2,h/2,w/2] [b,2,h/4,w/4]
            配准的三级结果。
        a_y : tensor.float32 [b,2,h,w]
            配准的标签，未分级。
        Returns
        -------
        loss_total : TYPE
            DESCRIPTION.
        """
        # Loss of deformation
        loss1= 0.
        for i in range(len(self.w_c)):
            loss1+= torch.mean(( y_rgted[:,i] - s_Y1[:,i]) ** 2) * self.w_c[i]
            
        loss_df = loss1 + self.gradloss(flow)
        
        # Loss of seg
        loss2= 0.
        for i in range(len(self.w_c)):
            loss2+= nn.BCELoss()(s_y[:,i],s_Y2[:,i]) * self.wc_hat[i]
            
        loss_seg=loss2
        
        # Loss of mask
        loss3= 0.
        change=self.getchange(y_rgted, s_y)
        loss_change=self.dice_loss_fake(change,s_Y1[:,-2:,...])
        #
        loss=10*loss_df + 5*loss_seg + 5*loss_change
        return loss
        
    def dice_loss(self,inp, target):  
        
        smooth = 1. # 避免黑图
        iflat = inp.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        
        dice=1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth)) 
        return dice
    
    def normalized_hamming_distance(self,img, lab):

        assert img.shape == lab.shape
        # binary
        imgbin=(img>0.5).float()
        labbin=(lab>0.5).float()
        # 计算汉明距离
        distance = (imgbin != labbin).sum()
        
        # 归一化距离
        normalized_distance = distance /img.numel()
        return normalized_distance 
    
    def loss_align(self,b_x,b_y,s_y,w_c):
        """
        Parameters
        ----------
        某一层的配准结果、配准标签、分割标签、类别系数
            加入分割标签是因为其有配准标签没有的像元类别信息
        Returns
        -------
        某一层的配准： 带有像元类别系数的均方根损失
            (进行了batch平均和pixel平均)
        """
        #c=len(w_c)
        m=b_x.size(0)
        #s_y=F.one_hot(s_y,c)#(b,h,w,4)
        s_y=s_y.permute(0,2,3,1)
        n_c=s_y.sum(0).sum(0).sum(0)#(4)
        coe=s_y*w_c/n_c
        coe=torch.where(torch.isnan(coe),torch.full_like(coe,0),coe)#类别数是0会出现NaN
        coe=coe.permute(0,3,1,2).sum(1)
        norm=((b_x-b_y)**2).sum(1)
        return (norm*coe).sum()/m
    
    def dice_loss_fake(self,inp, target):  
         
         smooth = 1. # 避免黑图
         iflat = inp.contiguous().view(-1)
         tflat = target.contiguous().view(-1)
         intersection = (iflat * tflat).sum()
         
         dice=1 - ((2. * intersection + smooth) /
                   (iflat.sum() + tflat.sum() + smooth)) 
         return dice
        
    def getchange(self,regt,pre):
        regt,pre=regt[:,0],pre[:,1]#[b,x,y]
        #print(regt.shape,pre.shape)
        onemat=torch.ones_like(regt)
        #regt=(regt>0.5)*onemat
        #pre=(pre>0.5)*onemat
        removed=regt*(onemat-pre)
        new=pre*(onemat-regt)
        out=torch.stack((removed,new),dim=1)
        return out#[b,2,x,y]


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self,  y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()
    

if __name__=='__main__':
    #inital
    dfpre=torch.randn((1,2,256,256))
    loss_g = Grad('l2',None).loss(dfpre)
    print(loss_g)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
 #%%   
    
    w_c=torch.tensor([0,0.1,1,10])
    wc_hat=torch.tensor([0.05,0.1,1,10])
    alpha=torch.tensor([0.5,0.35,0.15])
    
    
    
    
    

    crt=criterion(w_c, wc_hat, alpha)
    
    s1=torch.rand(size=(1,4,32,48))
    s2=torch.rand(size=(1,4,16,24))
    s3=torch.rand(size=(1,4,8,12))
    s_x=[s1,s2,s3]
    s_y=torch.randint(0,2,size=(1,6,32,48))
    
    a1=torch.randint(0,4,size=(1,2,32,48))
    a2=torch.randint(0,4,size=(1,2,16,24))
    a3=torch.randint(0,4,size=(1,2,8,12))
    a_x=[a1,a2,a3]
    a_y=torch.randint(0,4,size=(1,2,32,48))
    ax=torch.rand(size=(1,3,32,48))
    out=crt(ax,s_x,s_y.float(),a_x,a_y)
    print(out)
