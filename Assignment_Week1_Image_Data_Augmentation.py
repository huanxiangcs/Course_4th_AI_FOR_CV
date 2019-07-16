# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 20:30:57 2019

@author: PAUL
"""
import cv2
import random
import numpy as np
from matplotlib import pyplot as plt
from random import choice

#Crop Image
def img_crop(img, x1, y1, x2, y2): #input: a image, the coordinates of two points
    return img[x1:x2, y1:y2]

#Shift Color
def random_light_color(img):
    #get the brightness value
    B, G, R = cv2.split(img)
    
    b_rand = random.randint(-50, 50)
    if b_rand == 0:
        pass
    elif b_rand > 0:
        lim = 255 - b_rand
        B[B > lim] = 255
        B[B <= lim] = (b_rand + B[B <= lim]).astype(img.dtype)
    elif b_rand < 0:
        lim = 0 - b_rand
        B[B < lim] = 0
        B[B >= lim] = (b_rand + B[B >= lim]).astype(img.dtype)
        
    g_rand = random.randint(-50, 50)
    if g_rand == 0:
        pass
    elif g_rand > 0:
        lim = 255 - g_rand
        G[G > lim] = 255
        G[G <= lim] = (g_rand + G[G <= lim]).astype(img.dtype)
    elif g_rand < 0:
        lim = 0 - g_rand
        G[G < lim] = 0
        G[G >= lim] = (g_rand + G[G >= lim]).astype(img.dtype)
        
    r_rand = random.randint(-50, 50)
    if r_rand == 0:
        pass
    elif r_rand > 0:
        lim = 255 - r_rand
        R[R > lim] = 255
        R[R <= lim] = (r_rand + R[R <= lim]).astype(img.dtype)
    elif r_rand < 0:
        lim = 0 - r_rand
        R[R < lim] = 0
        R[R >= lim] = (r_rand + R[R >= lim]).astype(img.dtype)
    
    img_merge = cv2.merge((B, G, R))
    return img_merge

#Gamma Correction
def adjust_gamma(image, gamma):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i/255.0)**invGamma) * 255)
    table = np.array(table).astype('uint8')
    return cv2.LUT(img_dark, table)

#rotation
def img_rotation(img, angle, scale):
    M = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), angle, scale)
    img_rotate = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return img_rotate

#Perspective Transformation
def img_warp(img, pts1, pts2):
    
    height, width, channels = img.shape
    
    
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    M_warp = cv2.getPerspectiveTransform(pts1, pts2)       #generate the matrix
    img_warp = cv2.warpPerspective(img, M_warp, (width, height))
    return M_warp, img_warp

if __name__ == "__main__":
    img_address = input("请输入图片路径，(地址栏中用/代替'\\')：")
    img_transform = input("选择图片保存路径：")
    n = input("所需增强图像的数量:")
    img = cv2.imread(img_address)
    transforms = [random_crop,color_shift,rotation_img,random_warp]
    for i in range(int(n)):
        aug_img = random.choice(transfomrs)(img)
        cv2.imwrite("img_data/{}.jpg".format(i),aug_img)
    
