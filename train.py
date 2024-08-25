# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 20:32:35 2023

@author: Administrator
"""

import torch,os,time,copy,sys
import numpy as np
from torchvision import models,transforms
import matplotlib.pyplot as plt 
from scipy.interpolate import make_interp_spline
from metric1 import Acc_Vertex,calculate_iou

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"# OMP: Error #15
 
def xyspline(x,y):
    X_Y_Spline = make_interp_spline(x, y)
    X_ = torch.linspace(x.min(), x.max(), 500)
    Y_ = X_Y_Spline(X_)
    return X_,Y_

def dice_loss_fake(inp, target):  
     
     smooth = 1. # 避免黑图
     iflat = inp.contiguous().view(-1)
     tflat = target.contiguous().view(-1)
     intersection = (iflat * tflat).sum()
     
     dice=1 - ((2. * intersection + smooth) /
               (iflat.sum() + tflat.sum() + smooth)) 
     return dice


 
    
def getchange(regt,pre):
    regt,pre=regt[:,0],pre[:,1]#[b,x,y]
    #print(regt.shape,pre.shape)
    onemat=torch.ones_like(regt)
    #regt=(regt>0.5)*onemat
    #pre=(pre>0.5)*onemat
    removed=regt*(onemat-pre)
    new=pre*(onemat-regt)
    out=torch.stack((removed,new),dim=1)
    return out#[b,2,x,y]

def train_Seg_align(net, train_iter, test_iter,loss_func,
                   optimizer, num_epochs, device,sigma,path_log):
    print("training on",device)   
    #布置保存路径
    date=time.strftime('%Y%m%d%H')
    path_module=os.path.join(path_log,f'{date}')
    if not os.path.exists(path_module):
        os.mkdir(path_module)
    print(path_module)
    #
    train_loss_all=[]
    train_iou_all=[]
    train_acc_all,train_ac0_all=[],[]
    train_rmse_all,test_rmse_all=[],[]
    test_loss_all=[]
    test_iou_all=[]
    test_acc_all=[]
    best_loss=1e10
    #net = nn.DataParallel(net, device_ids=[0,1])
    net.to(device)
    best_model_wts=copy.deepcopy(net.state_dict()) #备份此时的参数
    since=time.time()
    for epoch in range(num_epochs):
        print('-'*10)
        print('Epoch{}/{}'.format(epoch+1,num_epochs))        
        train_num,test_num=0,0
        train_loss,train_iou,train_acc,train_ac0=0.0, 0.0 ,0.0,0.0
        test_loss, test_iou,test_acc=0.0 ,0.0 ,0.0
        train_mse, test_mse=0.0, 0.0
        net.train()
        for step,(s_x,s_Y,a_x,a_Y) in enumerate(train_iter):

            optimizer.zero_grad()
            s_x=s_x.to(device)
            a_x=a_x.to(device)
            s_Y=s_Y.float().to(device)
            a_Y=a_Y.float().to(device)
            #
            y_rgted, flow_pre, s_y=net(a_x,s_x,sigma)
            #
            loss=loss_func(y_rgted,s_Y[:,4:],s_y,s_Y[:,:4],flow_pre)
          

            loss.backward()
            optimizer.step()
            #评价
            mse=torch.mean((flow_pre - a_Y*sigma) ** 2)
            iou=calculate_iou(s_y[:,[1],...],s_Y[:,[1]])
            #
            train_mse+=mse.data.cpu()*s_x.size(0)#均方误差
            train_iou+=iou.item()*s_x.size(0)#iou
            train_loss+=loss.item()*s_x.size(0)
            train_num+=s_x.size(0)

        train_loss_all.append(train_loss/train_num)
        train_iou_all.append(train_iou/train_num)
        train_rmse_all.append(torch.sqrt(train_mse/train_num))#均方根误差
        print("Train-loss：{:.4f}".format(train_loss_all[-1]))
        print("iou on train：",(train_iou_all[-1]))
        print("rmse on train: {:.4f}".format(train_rmse_all[-1]))
        #print("acc on train Vertex：{:.4f}".format(train_acc_all[-1]))
        net.eval()
        for step,(s_x,s_Y,a_x,a_Y) in enumerate(test_iter):

            s_x=s_x.to(device)
            a_x=a_x.to(device)
            s_Y=s_Y.float().to(device)
            a_Y=a_Y.float().to(device)
            #
            y_rgted, flow_pre, s_y=net(a_x,s_x,sigma)
            #
            loss=loss_func(y_rgted,s_Y[:,4:],s_y,s_Y[:,:4],flow_pre)
         
            #评价
            mse=torch.mean((flow_pre - a_Y*sigma) ** 2)
            iou=calculate_iou(s_y[:,[1],...],s_Y[:,[1]])
            #
            test_mse+=mse.data.cpu()*s_x.size(0)#均方误差
            test_iou+=iou.item()*s_x.size(0)#iou
            test_loss+=loss.item()*s_x.size(0)
            test_num+=s_x.size(0)

        test_loss_all.append(test_loss/test_num)
        test_iou_all.append(test_iou/test_num)
        test_rmse_all.append(torch.sqrt(test_mse/test_num))#均方根误差
        print("Ttest-loss：{:.4f}".format(test_loss_all[-1]))
        print("iou on test：{}",(test_iou_all[-1]))
        print("rmse on test: {:.4f}".format(test_rmse_all[-1]))
        #print("acc on train Vertex：{:.4f}".format(train_acc_all[-1]))
    
        ##可视化损失
        plt.figure(num=0)
        plt.plot(train_loss_all,color='r', linestyle='--', label='train')
        plt.plot(test_loss_all,color='b', linestyle='--', label='test')
        plt.legend(loc='best')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Curve of Loss Change with epoch')
        #可视化iou
        plt.figure(num=1)
        plt.plot(train_iou_all,color='r', linestyle='--', label='train_iou')
        plt.plot(test_iou_all,color='b', linestyle='--', label='test_iou')
        plt.legend(loc='best')  # 提供11种不同的图例显示位置
        plt.xlabel('epoch')
        plt.ylabel('ratio')
        plt.title('Curve of metric Change with epoch')
        #可视化rmse
        plt.figure(num=2)
        plt.plot(train_rmse_all,color='r', linestyle='--', label='train_rmse')
        plt.plot(test_rmse_all,color='b', linestyle='--', label='test_rmse')
        plt.legend(loc='best')  # 提供11种不同的图例显示位置
        plt.xlabel('epoch')
        plt.ylabel('ratio')
        plt.title('Curve of RMSE Change with epoch')
        plt.show()

        
        ##保存损失和模型
        over=time.time()
        print(f'train until current epoch Timeuse:{(over-since)//60}min,{int((over-since)%60)}s')
        #train_acc_all.append(train_acc.double().item()/train_num/(s_x.size(-1))/(s_x.size(-2)))        
        torch.save(net,os.path.join(path_module,f'epoch{epoch+1}.pth'))
        np.save(os.path.join(path_module,'train_loss.npy'),np.array(train_loss_all)) 
        np.save(os.path.join(path_module,'train_iou.npy'),np.array(train_iou_all)) 
        np.save(os.path.join(path_module,'train_rmse.npy'),np.array(train_rmse_all))
#        np.save(os.path.join(path_module,'train_acc.npy'),np.array(torch.stack((train_acc_all),dim=0)))
        np.save(os.path.join(path_module,'test_loss.npy'),np.array(test_loss_all)) 
        np.save(os.path.join(path_module,'test_iou.npy'),np.array(test_iou_all)) 
        np.save(os.path.join(path_module,'test_rmse.npy'),np.array(test_rmse_all))
#        np.save(os.path.join(path_module,'test_acc.npy'),np.array(torch.stack((test_acc_all),dim=0)))

        #验证
        if test_loss_all[-1] < best_loss:
            best_loss = test_loss_all[-1]
            best_model_wts=copy.deepcopy(net.state_dict())
    #net.load_state_dict(best_model_wts)
    print(test_loss_all.index(best_loss))
        
    