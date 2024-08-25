# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 17:12:57 2023

@author: jwl
"""

import numpy as np
from sklearn.metrics import confusion_matrix as CM
#import sklearn
class Evaluator():
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)
        self.eps = 1e-8

    def get_tp_fp_tn_fn(self):
        tp=self.confusion_matrix[1,1]
        fp=self.confusion_matrix[0,1]
        tn=self.confusion_matrix[0,0]
        fn=self.confusion_matrix[1,0]
        return tp, fp, tn, fn

    def Precision(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        precision = tp / (tp + fp + self.eps)
        return precision

    def Recall(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        recall = tp / (tp + fn + self.eps)
        return recall

    def F1(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Precision = tp / (tp + fp + self.eps)
        Recall = tp / (tp + fn + self.eps)
        F1 = (2.0 * Precision * Recall) / (Precision + Recall + self.eps)
        return F1

    def OA(self):
        OA = np.diag(self.confusion_matrix).sum() / (self.confusion_matrix.sum() + self.eps)
        return OA 

    def IoU(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        IoU = tp / (tp + fn + fp + self.eps)
        return IoU

    def Dice(self):
        tp, fp, tn, fn = self.get_tp_fp_tn_fn()
        Dice = 2 * tp / ((tp + fp) + (tp + fn))
        return Dice

    def _generate_matrix(self, gt_image, pre_image):
        confusion_matrix=CM(gt_image.flatten(),pre_image.flatten())
# =============================================================================
#         mask = (gt_image >= 0) & (gt_image < self.num_class)
#         label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
#         count = np.bincount(label, minlength=self.num_class ** 2)
#         confusion_matrix = count.reshape(self.num_class, self.num_class)
# =============================================================================
        return confusion_matrix#np.rot90(confusion_matrix, k=2).T#改动

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape, 'pre_image shape {}, gt_image shape {}'.format(pre_image.shape,
                                                                                                 gt_image.shape)
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

     
if __name__ == '__main__':

    gt = np.array([[[0, 1, 1],
                   [1, 1, 1],
                   [1, 0, 1]]])

    pre = np.array([[[0, 0, 1],
                   [1, 0, 1],
                   [1, 1, 1]]])

    eb = Evaluator(num_class=2)
    eb.add_batch(gt, pre)
    print(eb.confusion_matrix)
    print(eb.get_tp_fp_tn_fn())
    print(eb.Precision())#准确率
    print(eb.Recall())#召回率
    print(eb.IoU())#交并比
    print(eb.OA())#Overall Accuracy精确度
    print(eb.F1())#F1分数，准确率和召回率算得
    
    # 举例

