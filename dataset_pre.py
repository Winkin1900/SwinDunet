# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 12:00:34 2023

@author: Administrator
"""


# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 18:41:35 2023
#要求对样本和标签进行：随机亮度、对比度、饱和度(只对样本)
                    随机旋转、水平反转、裁剪（需要一一对应）
有关Colorjitter：做0.5即50%的偏移摆动
#torchvision似乎只能读jpg,png,没法读tif。cv可以。
新建和拆除的返回地址没变。记得copy
@author: l
"""
import torch,os,sys,time,random
import cv2 as cv
import numpy as np
from torchvision import transforms
import matplotlib.pyplot  as plt

sys.path.append(r'.\dataset\utils')
import drop_polygon as dpn
import add_polygon as apn
import generate_deformation as gdf
import apply_deformation



def create_dxm(gray):
    """
    对单通道标签生成点线面多通道图层
    """
    # 轮廓检测
    h,w=gray.shape
    blank = np.zeros((h,w), dtype=np.uint8)
    contours,_=cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(blank, contours, -1, 255,1)  #线在面上（边界）
    #line=cv.Canny(gray,100,200)#线
    points=np.zeros_like(gray)#点检测
    corners = cv.goodFeaturesToTrack(gray,0,0.5,4)
    if corners is not None:#进行角点检测的label中，在没有角点的情况下返回时全黑Nonetype,要么执行判断，要么删减数据集。
        corners = np.intp(corners)
        for i in corners:
            x,y = i.ravel()
            points[y,x]=255        
    #拼接（背景，面，线，点）
    backg=points+blank+gray
    backg=np.where(backg>0,0,255)
    dxm=np.stack((backg,gray,blank,points),0)  
    return dxm #(4,h,w)np.U8

def array2tensor(img):
    img=torch.from_numpy(img.astype(np.float32))/255
    t=img.unsqueeze(0).permute(0,3,1,2)
    return t#(1,c,h,w)

def read_whan3_images(path,is_train=True,sigma=4):#返回列表，元素尺寸（512，512，3）
    """
    样本和标签读到列表
    """
    path=os.path.join(path,'train' if is_train else 'test')
    names=os.listdir(os.path.join(path,'label'))
    names.sort(key=len)
    feas_seg,label_seg,feas_align,label_align=[],[],[],[]
    size=(256,256)
    STN=apply_deformation.SpatialTransformer(size)
    center_crop=transforms.CenterCrop(size)
    for i,fname in enumerate(names):
        
        image=cv.imread(os.path.join(path,'image',f'{names[i]}'),1)
        label=cv.imread(os.path.join(path,'label',f'{names[i]}'),1)
        misMap=cv.imread(os.path.join(path,'misMap',f'{names[i]}'),0)
        
        gray=cv.imread(os.path.join(path,'label',f'{names[i]}'),0)
        gray=np.array(center_crop(torch.from_numpy(gray)))
        # 生成feas_seg列表
        fs=cv.imread(os.path.join(path,'image',f'{names[i]}'),1)
        feas_seg.append(np.array(center_crop(torch.from_numpy(fs).permute(2,0,1)).permute(1,2,0)))#(h,w,3)255U8
        #                      
        #上移增加判定
        img=create_dxm(gray)
        drop_gray,n = dpn.rand_drop_polygon(gray.copy(),0.1)
        #
        new=gray-drop_gray #
        add_drop_gray=apn.rand_add_building(drop_gray.copy(),building_num=n)
        AD_gray=torch.from_numpy(add_drop_gray.astype(np.float32))/255
        removed=add_drop_gray-drop_gray
        img_ad=create_dxm(add_drop_gray)
        
        # 合成label_seg列表
        img=np.concatenate((img,img_ad[[1,2,3]],removed[np.newaxis,...],new[np.newaxis,...]),axis=0)
        img=torch.tensor(img,dtype=torch.float32)/255
        label_seg.append(img[:,...])#(4,h,w)
        
        # 应用形变
        df,max_df=gdf.G_df_with_param(size,sigma),sigma
        src=img_ad[[1,2,3],...].transpose(1,2,0)#(h,w,3)U8np
        src,df=array2tensor(src) ,array2tensor(df*255)#抵消一下
        joint_sd=torch.cat((src,df),dim=1)
        warp=STN(joint_sd,df)
        src,df = warp[:,[0,1,2]], -warp[:,[3,4]]
        # 合成label_align列表
        df=df.squeeze(0)/max_df
        label_align.append(df)#df.squeeze(0)/max_df)#(2,h,w)
        # 合成feas_align列表
        feas_align.append(src.squeeze(0))#(3,h,w)
        #存
        if i==100:
            break
        #print(i)
        
    return feas_seg,label_seg,feas_align,label_align

def rand_rotate_crop(fea_seg, label_seg, fea_align, label_align, size=(256,256)):
    """
    对数据集的图像进行旋转和裁剪
    Parameters
    ----------
    feature label : tensor(0-1)[c,h,w]
    size : TYPE, optional
        DESCRIPTION. The default is (360,360).
    Returns
    -------
    feature label : tensor(0-1)[c,h,w]
    """
    angle=random.randint(0,180)
    center_crop=transforms.CenterCrop(size)
    #print(angle)
    fea_seg = center_crop(fea_seg)
    label_seg = center_crop(label_seg)
    fea_align= center_crop(fea_align)
    label_align = center_crop(label_align)
    #label_align = center_crop(transforms.functional.rotate(label_align, angle))
    return fea_seg, label_seg, fea_align, label_align

class Whan3Dataset(torch.utils.data.Dataset):
    def __init__(self,path,is_train,sigma):
        """transforms.ToTensor可以一步实现：
        转类型：numpy->tensor
        变通道：(h,w,3)->(3,h,w)
        转精度：U8->float32
        变范围：(0-255)->(0-1)
        """
        self.transform=transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomResizedCrop((200, 200), scale=(0.1, 1), ratio=(0.5, 2)),
                transforms.ColorJitter(brightness=0.5, contrast=0, saturation=0, hue=0),
                ])
        self.feas_seg,self.label_seg,self.feas_align,self.label_align=read_whan3_images(
            path,is_train,sigma)
        self.feas_seg=[self.transform(feature) for feature in self.feas_seg]
        print('read ' + str(len(self.feas_seg)) + ' examples')
   
    def __getitem__(self,idx):
        a=self.feas_seg[idx]
        b=self.label_seg[idx]
        c=self.feas_align[idx]
        d=self.label_align[idx]
        #a,b,c,d=rand_rotate_crop(a,b,c,d)
        return (a,b,c,d)
    
    def __len__(self):
        return len(self.feas_seg)
        
def load_data_Whu3(batch_size,path,sigma,num_workers=0):
    """
    从Dataset返回Dataloader
    parameters:
        batch_size
        path
        num_workers
        z:最大变形位置
    """
    train_iter=torch.utils.data.DataLoader(
        Whan3Dataset(path,True,sigma),batch_size,shuffle=False,
        drop_last=True,num_workers=num_workers)
    test_iter=torch.utils.data.DataLoader(
        Whan3Dataset(path,False,sigma),batch_size,
        drop_last=True,num_workers=num_workers)
    return train_iter,test_iter
 
if __name__=='__main__':
    
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    since=time.time()
    #path=r"D:\Ajwl\data\Whan3_shuffle833\tiles_Images"
    #path=r'D:\Ajwl\data\MassaDataset'
    path=r'D:\Ajwl\data\whu3_origin\w3'
    #train_iter,test_iter=load_data_Whu3(4,path)
    sigma=4.
    dataset1=Whan3Dataset(path,True,sigma)
    #dataset2=Whan3Dataset(path,False,sigma)
    over=time.time()
    print(f'Timeuse:{(over-since)//60}min,{int((over-since)%60)}s')
    #%%
    i=6
    a,b,c,d=dataset1[i]
    m=[a,b,c,d]
    for j in range(4):
        print(m[j].shape)
    for j in range(4):
        print(m[j].max(),m[j].min())     



    

        
        
         
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        