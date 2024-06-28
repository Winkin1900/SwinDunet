# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:53:16 2023
对于Arial数据集，buidingsize=10, randxy=[5-10]
对于Whu数据集，35,20-50
@author: l
"""

import numpy as np
import cv2

def rand_add_building(image, building_size=35,building_num=1):
    """
    building_size 是缓冲区半径 也是随机产生成矩形的最大质心臂长。 35=(50)/2*根号2
    """
    
    while building_num >0:
        non_zero_pixels = cv2.findNonZero(image)
        # 如果原始图像中没有建筑物，直接添加新建筑物
        if non_zero_pixels is None:
            return image
        # 创建一个掩膜图像，标记已有建筑物的区域
        mask = np.zeros_like(image, dtype=np.uint8)
        for pixel in non_zero_pixels:
            cv2.circle(mask, (pixel[0][0], pixel[0][1]), building_size,255, -1)
        # 生成随机位置的坐标

        #
        zero_pixels = cv2.findNonZero(255-mask)
        if zero_pixels is None:
            return image
        building_num-=1
        rand_center=np.random.randint(0,zero_pixels.shape[0])
        #print(rand_center,zero_pixels[rand_center,0,0],zero_pixels[rand_center,0,1])
        add_rotated_rectangle(image,zero_pixels[rand_center,0,0].astype(np.float32),zero_pixels[rand_center,0,1].astype(np.float32))

    return image #(h,w)

def add_rotated_rectangle(image,center_x,center_y):

    # 生成随机大小的矩形
    rectangle_width = np.random.randint(20,50)
    rectangle_height = np.random.randint(20,50)
    # 生成随机的旋转角度 
    angle_deg = np.random.randint(0, 180)
    #angle_rad = math.radians(angle_deg)#转弧度
    # 创建旋转矩阵 
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle_deg, 1)
    # 计算矩形的四个角点
    rect_points = np.array([[-rectangle_width / 2 + center_x, -rectangle_height / 2 + center_y],
                            [rectangle_width / 2 + center_x, -rectangle_height / 2+ center_y],
                            [rectangle_width / 2+ center_x, rectangle_height / 2+ center_y],
                            [-rectangle_width / 2+ center_x, rectangle_height / 2+ center_y]])
    # 对角点进行旋转变换
    rotated_rect_points = np.dot(rotation_matrix[:, :2], rect_points.T).T + rotation_matrix[:, 2]
    # 转换为整数坐标
    rotated_rect_points = rotated_rect_points.astype(np.int32)
    # 在图像上绘制旋转后的矩形
    cv2.drawContours(image, [rotated_rect_points], 0, 255, -1)
    return image

if __name__=='__main__':
    #image = np.zeros((image_height, image_width), dtype=np.uint8)
    image=cv2.imread(r'D:\Ajwl\data\Whan3_shuffle833\tiles_Images\test\label\2.tif',0)
    image0=image.copy()
    # 在空白位置随机添加新建筑物
    building_size = 20  
    rand_add_building(image,building_num=10)
    #print(id(image0),id(image_with_building))
    # 显示图像
    cv2.imshow('Iamge',image0)
    cv2.imshow("Image with Buildings", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
