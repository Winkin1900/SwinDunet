# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 12:11:16 2024

@author: jwl
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 10:59:09 2024

@author: jwl
"""
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from apply_deformation import SpatialTransformer

# 生成网格坐标
def generate_grid(shape):
    rows, cols = shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))
    return np.stack((x, y), axis=-1)

# 生成形变场
def generate_deformation_field(size, v0, v_list, x_list, S_list):
    grid = generate_grid(size)
    deformation_field = np.zeros_like(grid, dtype=np.float32)
    for i in range(len(v_list)):
        v_i = v_list[i]
        x_i = x_list[i]
        S_i = S_list[i]
        
        diff = grid - x_i
        exponent = -np.sum(diff @ S_i * diff, axis=-1)
        deformation_field += v_i * np.exp(exponent)[..., np.newaxis]
    
    deformation_field = v0 + deformation_field
    return deformation_field



# flow
def geneate_quv(phi,c=15):
    plt.figure(0,dpi=600)
    x = np.linspace(-2, 2, c)  # x坐标范围和点的数量
    y = np.linspace(-2, 2, c)  # y坐标范围和点的数量
    X, Y = np.meshgrid(x, y)  # 创建网格
    U = cv2.resize(phi[:,:,0],(c,c))*1 # x分量
    V = cv2.resize(phi[:,:,1],(c,c))*1# y分量
    #M = np.hypot(U, V)  # 向量长度 来表征颜色映射的
    # 绘制向量场，使用M作为颜色映射s
    plt.quiver(X, Y, U, V)
    #plt.colorbar() 
    plt.axis('off')
    plt.show()

def G_df_with_param(size,sigma):
    # param
    v0 = np.array([sigma,sigma], dtype=np.float32)  # 设置较小的全局平移
    num_gaussians = 2  # 使用较少的高斯函数
    v_list = [np.random.uniform(-3, 3, 2) for _ in range(num_gaussians)]  # 较小的随机平移
    x_list = [np.random.uniform([0, 0], size, 2) for _ in range(num_gaussians)]
    S_list = [np.diag(np.random.uniform(0.00001, 0.00005, 2)) for _ in range(num_gaussians)]  # 较小的协方差    
    deformation_field = generate_deformation_field(size, v0, v_list, x_list, S_list)
    return deformation_field
    
def array2tensor(img):
    img=torch.from_numpy(img.astype(np.float32))
    if len(img.shape)==3:
        t=img.unsqueeze(0).permute(0,3,1,2)
    elif len(img.shape)==2:
        t=img.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("shape error")
    return t#(1,c,h,w)


if __name__=='__main__':
    
    # 参数设置
    image = cv2.imread('C:/Users/Administrator/Pictures/Camera Roll/4-24V/1001-2.png', 0)
    rows, cols = image.shape
    #image = np.array(image, dtype=np.float32)
    
    deformation_field=G_df_with_param( image.shape,2.)
    print(deformation_field.max())
    
    # cal avg
    sum_df=[]
    for i in range(100):
        deformation_field=G_df_with_param(image.shape,2.)
        sum_df.append(deformation_field.max())
    sdf=np.array(sum_df)
    print('avg:',sdf.mean())
    
    # 应用形变
    
    STN=SpatialTransformer(image.shape)
    deformed_image =STN(array2tensor(image), array2tensor(deformation_field))
    geneate_quv(deformation_field)
    warp= np.array(deformed_image.squeeze(0).squeeze(0),dtype=np.uint8)
    # 显示结果
    cv2.imshow('image',image)
    cv2.imshow('warp',warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
