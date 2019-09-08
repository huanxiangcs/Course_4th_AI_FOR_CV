# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 13:55:42 2019

@author: PAUL
"""
import numpy as np

def myNMS(lists, thre):
    
    result_idx = [] #初始化索引值集合
    
    x1 = lists[:, 0]
    y1 = lists[:, 1]
    x2 = lists[:, 2]
    y2 = lists[:, 3]
    
    area = (x2 - x1 + 1) * (y2 -y1 + 1)
    
    scores = lists[:, 4]
    order = scores.argsort()[::-1] #降序排列 返回索引值
    
    while len(order) > 0:
        idx = order[0] #取scores最大值索引
        result_idx.append(idx) #加入到要返回的索引值集合
        
        #IOU计算，索引减一
        iou_x1 = np.maximum(x1[idx], x1[order[1:]])
        iou_y1 = np.maximum(y1[idx], y1[order[1:]])
            
        iou_x2 = np.minimum(x2[idx], x2[order[1:]])
        iou_y2 = np.minimum(y2[idx], y2[order[1:]])
            
        w = np.maximum(0, iou_x2 - iou_x1 + 1)
        h = np.maximum(0, iou_y2 - iou_y1 + 1)
            
        iou_area = w * h
            
        iou = iou_area / (area[idx] + area[order[1:]] - iou_area)
            
        idx_iou = np.where(iou < thre)[0] #选取要保留的索引值
            
        idx_order = idx_iou + 1 #将索引值对应到order内
            
        order = order[idx_order]

        
    return result_idx

if __name__ == "__main__":
    
    thre = 0.1
    lists = np.zeros((10, 5))
    lists[:, 0:4] = np.random.randint(100, size =[10, 4])
    lists[:, 4]   = np.random.rand(10)
    
    idx_row = myNMS(lists, thre)
    print(lists[idx_row])
    