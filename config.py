# -*- coding: utf-8 -*-
"""
Created on Tue May  9 19:38:37 2023
工程的总控制，链接所有API
@author: zh
"""

import torch,time,os,random,sys
import numpy as np  
import dataset_pre as Dp
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Fixed random number. careful
def init_seeds(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
init_seeds(3228)

# set GPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
for idx in range(torch.cuda.device_count()):
    print(idx,torch.cuda.get_device_name(idx))

#%% load dataset
since=time.time()
#path=r".\Simulated_Whu3\o.1DAR"
path=r'D:\Ajwl\data\whu3_origin\w3'
batch_size,sigma = 16,4 # maximum of df
train_iter,test_iter=Dp.load_data_Whu3(batch_size,path,sigma,num_workers=0)
over=time.time()
print(f'read data Timeuse:{(over-since)//60}min,{int((over-since)%60)}s')

#%% train
from train import train_Seg_align   
from networks import SwinDunet
from criterion import criterion

#model
module=r'.\logs\log_name'
if os.path.exists(module):
    net=torch.load(module)
    print("Pre-trained model loaded") 
else:
    net=SwinDunet(inshape=(256,256))
    print("General initialization parameters")
# loss func
w_c=torch.tensor([10,10,10])
wc_hat=torch.tensor([0.05,0.1,1,10])
alpha=torch.tensor([0.5,0.35,0.15])
print(device)
loss_func=criterion(w_c.to(device), wc_hat.to(device), alpha.to(device),sigma)

# optimizer
optimizer=torch.optim.Adam(net.parameters(),lr=0.001,weight_decay=1e-4)
#其他
num_epochs=100
path_log=r".\logs"
tain1=train_Seg_align(net,train_iter,test_iter,loss_func,
                        optimizer,num_epochs,device,sigma,path_log)

