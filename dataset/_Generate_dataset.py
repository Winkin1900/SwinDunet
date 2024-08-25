# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:38:57 2024

@author: Administrator
"""

import torch,os,sys,time,random
import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot  as plt

sys.path.append(r'.\utils')
import drop_polygon as dpn
import add_polygon as apn
import generate_deformation
import apply_deformation


def array2tensor(img):
    img=torch.from_numpy(img.astype(np.float32))
    if len(img.shape)==3:
        t=img.unsqueeze(0).permute(0,3,1,2)
    elif len(img.shape)==2:
        t=img.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("shape error")
    return t#(1,c,h,w)


def Generate_data(path, path_save, DAR=0.1, is_train=True, sigma=4., size=(256,256)):
    # path
    path=os.path.join(path,'train' if is_train else 'test')
    path_save=os.path.join(path_save,'train' if is_train else 'test')
    if not os.path.exists(path_save):
        os.makedirs(os.path.join(path_save,'image'))
        os.makedirs(os.path.join(path_save,'label'))
        os.makedirs(os.path.join(path_save,'misMap'))
        
    # read
    names=os.listdir(os.path.join(path,'label'))
    names.sort(key=len)
    
    #df and crop
    STN= apply_deformation.SpatialTransformer(size)
    center_crop=transforms.CenterCrop(size)
    for i,fname in enumerate(names):
        #
        gray=cv.imread(os.path.join(path,'label',f'{names[i]}'),0)
        rgb=cv.imread(os.path.join(path,'image',f'{names[i]}'),1)
        gray=np.array(center_crop(torch.from_numpy(gray)))
        rgb=np.array(center_crop(torch.from_numpy(rgb).permute(2,0,1)).permute(1,2,0))
        
        # drop and add
# =============================================================================
#         if i%2==0:
#             add_drop_gray,drop_gray=gray,gray
#             drop_gray,n = dpn.rand_drop_polygon(gray.copy(),DAR)
#             new=gray-drop_gray #
#             add_drop_gray=apn.rand_add_building(drop_gray.copy(),building_num=n)
#             removed = add_drop_gray-drop_gray
#             
#         else:
#             add_drop_gray,drop_gray=gray,gray
# =============================================================================
        add_drop_gray,drop_gray=gray,gray
        drop_gray,n = dpn.rand_drop_polygon(gray.copy(),DAR)
        new=gray-drop_gray #
        add_drop_gray=apn.rand_add_building(drop_gray.copy(),building_num=n)
        removed = add_drop_gray-drop_gray
        # df 
        df= generate_deformation.G_df_with_param(size)
        src,df = array2tensor(add_drop_gray) ,array2tensor(df)#抵消一下
        warp_gray = STN(src, df)
        warp_gray = np.array(warp_gray.squeeze(0).squeeze(0),dtype=np.uint8)
        
        # label
        label =np.stack((drop_gray,gray,add_drop_gray),axis=2)
        
        # save
        cv.imwrite(os.path.join(path_save,'image',f'{names[i]}'),rgb)
        cv.imwrite(os.path.join(path_save,'label',f'{names[i]}'),label)
        cv.imwrite(os.path.join(path_save,'misMap',f'{names[i]}'),warp_gray)
        
# =============================================================================
#         #for try
#         print(i)
#         if i==5:
#             break
# =============================================================================
    return 0


if __name__=='__main__':
    
    #
    path=r'D:\Ajwl\data\AerialImageDataset\ViennaBig'
    save_path=r'..\Simulated_AerialImageDataset(Vienna)\o.1DAR'
    Generate_data(path,save_path,DAR=0.1, is_train=False)