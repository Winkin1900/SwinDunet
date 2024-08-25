# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:50:16 2023

@author: Administrator
"""

#!/usr/bin/python3
"""
iou有问题：outs被认定为了布尔值01
acc也有问题：outa被认定为0-4，没有缩放。
"""
import time,os,torch
import torch.nn as nn 
import numpy as np
from net_utils import metric_CM as cm



def vertex_mse(point,df,df_lab):
    #print(point.shape,df.shape,df_lab.shape)
    point=torch.ones_like(point)*(point>0.5)#[b,1,256,256]
    n=point.sum()
    
    diff=(((df-df_lab) ** 2).sum(1))
    mse=(diff*point[:,0]).sum()/n
    #print(mse)
    #mse= torch.mean(((df-df_lab) ** 2)*point)
    #print(mse)
    #print(torch.sqrt(mse))
    if torch.isnan(mse):
        mse=torch.tensor(0.)
    return mse
    
    
    
def IoU_seg(outs,s_y):
    """
    outs:list(3) (b,4,h,w) tensor.float32
    输出分割结果
    s_y: (b,4,h,w) tensor.float32
    分割标签
    return:
    分割的交并比（进行了batch平均）
    """ 
    m=s_y.size(0)
    #img_x=sx_out[0][:,1,:,:]#(b,h,w)
    s_x=outs
    #print(s_x.shape,s_y.shape)
    #s_x=s_x[:,1]#(b,h,w)
    img_x=torch.where(s_x>0.5,1,0)
    img_y=s_y#[:,1,:,:]#(b,h,w)
    eb = cm.Evaluator(num_class=2)
    eb.add_batch(np.array(img_y.cpu()), np.array(img_x.cpu()))
    eb.confusion_matrix
    eb.get_tp_fp_tn_fn()
    eval_cm=(eb.Precision(),eb.Recall(),eb.IoU(),eb.OA(),eb.F1())#准确率#召回率#交并比#Overall Accuracy精确度#F1分数，准确率和召回率算得
    return np.array(eval_cm)
# =============================================================================
# #%% 自定义iou
#     m=s_y.size(0)
#     #img_x=sx_out[0][:,1,:,:]#(b,h,w)
#     s_x=outs[0]
#     s_x=s_x[:,1]#(b,h,w)
#     img_x=torch.where(s_x>0.5,1,0)
#     img_y=s_y[:,1,:,:]#(b,h,w)
#     a=abs(img_x-img_y)
#     n1=a.sum(1).sum(1)
#     b=((img_x+img_y)-abs(img_x-img_y))/2
#     n2=b.sum(1).sum(1) 
#     iou=n2/(n1+n2)#(b,1)
#     Siou=iou.sum()/m
#     if torch.isnan(Siou):
#         Siou=torch.tensor(1.0)
#     return Siou
# =============================================================================

def Acc_Vertex(ax_out,a_y,s_y,t):
    """
    a_x:list(3) (b,2,h,w) tensor.float32
        输出配准结果
    a_y: (b,2,h,w) tensor.float32，已经归一化
        配准标签
    s_y: (b,4,h,w) tensor.float32
        分割标签，带有角点信息 
    t:阈值：最大容忍位移
    return：tensor.float32
    满足最大欧式距离（进行了batch平均），比例
    """
    vertex=s_y[:,2,:,:]#(b,h,w)
    #vertex.unsqueeze(1)#(b,1,h,w)
    Vacc,Vac0,mse=torch.zeros(13),torch.zeros(13),torch.tensor(0.)
    if not vertex.sum()==0:
        a_X=ax_out[0]
        aX0=torch.zeros_like(a_X)#创造0对照,平行计算
        norm, nrm0=(a_X-a_y)**2, (aX0-a_y)**2#(b,2,h,w)
        mse=(norm.sum(1)*vertex).sum()/vertex.sum()#均方根
        norm, nrm0=torch.sqrt(norm.sum(1)), torch.sqrt(nrm0.sum(1))#(b,h,w)
        acc=(norm*vertex).flatten()
        ac0=(nrm0*vertex).flatten()
        acc=acc[acc!=0]#不是顶点像元和误差=0 的会被扔掉。
        ac0=ac0[ac0!=0]
        for i in range(13):
            fcc=torch.where(acc>i,1,0)#超出误差的像元
            fra=1-fcc.sum()/vertex.sum()#反 比例就包含了误差为0像元
            Vacc[i]=fra
            fc0=torch.where(ac0>i,1,0)#模仿
            fra0=1-fc0.sum()/vertex.sum()
            Vac0[i]=fra0
            
    return [Vacc,Vac0,mse]#一维张量

def calculate_iou(tensor1, tensor2):
    # Flatten the tensors and calculate intersection and union
    tensor1=(tensor1>0.5).float()
    tensor2=(tensor2>0.5).float()
    intersection = torch.sum(tensor1 * tensor2)
    union = torch.sum(tensor1) + torch.sum(tensor2) - intersection

    # Avoid division by zero
    epsilon = 1e-6

    # Calculate IoU
    iou = intersection / (union + epsilon)

    return iou


if __name__ =='__main__': 
    s1=torch.randn(size=(1,4,32,48))
    s2=torch.randn(size=(1,3,16,24))
    s3=torch.randn(size=(1,3,8,12))
    s_x=[s1,s2,s3]
    s_y=torch.randint(0,2,size=(1,4,32,48))
    
    a1=torch.randint(0,4,size=(1,2,32,48))
    a2=torch.randint(0,4,size=(1,2,16,24))
    a3=torch.randint(0,4,size=(1,2,8,12))
    a_x=[a1,a2,a3]
    a_y=torch.randint(0,4,size=(1,2,32,48))
    
    m=torch.ones(1,2,32,48)
    m1=[m,m,m]
    n=m+1
    a=Acc_Vertex(m1,a_y,s_y.float(),4)
    print(a)
    b=IoU_seg(s_x, s_y.float())
    print(b)
    