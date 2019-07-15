# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 21:05:20 2019

@author: PAUL
"""
# Coding 1
import cv2
import numpy as np

def medianBlur(img, kernel, padding_way):
    
    H = img.shape[0] 
    W = img.shape[1]
    n = len(kernel[0])
    m = len(kernel[1])
    j = int(n/2) # COLS NUM TO BE PADDED TO THE LEFT AND RIGHT SIDE
    i = int(m/2) # ROWS NUM TO BE PADDED TO THE TOP AND BOTTOM 
    finalImg = np.zeros((H, W), dtype = int) # INITIALIZE THE FINAL BLURRED IMG
    paddedImg = np.zeros((H + 2*j, W + 2*i), dtype = int) #THE PADDED IMG


    if padding_way == 'REPLICA':
        
        for h in range(H): # PAD THE LEFT AND RIGHT SIDE FIRST
            leftValue = img[h, 0]
            listLeftValue = [leftValue] * int(m/2)
            rightValue = img[h, W-1]
            listRightValue = [rightValue] * int(m/2)
            
            paddedList = listLeftValue[:] 
            paddedList.extend(img[h, :])
            paddedList.extend(listRightValue[:])
            
            paddedImg[j+h, :] = paddedList
        
        #THEN PAD THE TOP AND BOTTOM
        upPadded = [paddedImg[j, :]] * int(n/2)
        downPadded = [paddedImg[j+H-1, :]] * int(n/2)
        
        upPadded.extend(paddedImg[j:j+H, :])
        upPadded.extend(downPadded[:])
        
        paddedImg = upPadded[:]
        
    elif padding_way == 'ZERO':
        for h in range(H):
            leftValue = 0
            listLeftValue = [leftValue] * int(m/2)
            rightValue = 0
            listRightValue = [rightValue] * int(m/2)
            
            paddedList = listLeftValue[:] 
            paddedList.extend(img[h, :])
            paddedList.extend(listRightValue[:])
            
            paddedImg[j+h, :] = paddedList
        
        upPadded = [np.zeros((W + 2*i), dtype = int)] * j
        downPadded = [np.zeros((W + 2*i), dtype = int)] * j
        
        upPadded.extend(paddedImg[j:j+H, :])
        upPadded.extend(downPadded[:])
        
        paddedImg = upPadded[:]
    
    for h in range(H):
        for w in range(W):
            convList = [] #STORE THE CONVOLUTION RESULT TEMPORARILY
            for N in range(n):
                for M in range(m):
                    value = paddedImg[h+N][w+M]
                    convList.extend([value])
            convList.sort() #SORT OBTAINED VALUE TO FIND THE MEDIAN
            finalImg[h, w] = convList[int(n*m/2)] #FIND THE MEDIAN
            
    
    return finalImg
if __name__ == '__main__':
    img = cv2.imread("C:/Users/PAUL/Documents/ML/CV Training course/20190706/lenna.jpg",0)
    cv2.imshow('lenna', img)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()

    kernel = np.zeros((5,5), dtype = int)
    padding_way = 'ZERO'
    medianBlurImg = medianBlur(img, kernel, padding_way)
    medianBlurImg = np.uint8(medianBlurImg)
    cv2.imshow('Median_Blur', medianBlurImg)
    key = cv2.waitKey()
    if key == 27:
        cv2.destroyAllWindows()
            
'''
Follow up1: Can it be completed in a shorter time complexity?

The current time complexity is H*W*(m*n)*log(m*n), which H*W comes from the two\
for in program and the (m*n)*log(m*n) is the complexity of built-in function \
.sort(). If the step size is 1 for the convolution, the current complexity seems\
to be the shortest.  
'''

#Coding 2
'''
Pseudo Code

Given:
    Points pairs from A and B to be classified.
    
    
Return:
    Final homography matrix
    
#the pseudo code below follows the process described in the assignment 2,
#however, the process of RANSAC I searched on Internet is different.
def ransacMatching(A, B):    
    iterations = 0
    #epsilon is given by experience
    epsilon - A threshold to classify points pairs into inliers or outliers.
    Maximum iterations k = log(0.01)/log(1-(N1/N)**4) #Follow up 1
    #N1/N is the proportion of inliers in the whole data set
    maybeInliers set = 4 pairs randomly selected form given pairs
    number of inliers n
    H calculated by the maybeInliers set
    while iterations < k: 
    
        flag of change in n = 0
        for maybeOutlier(AiBj) in other maybeOutliers:
            if ||Bj, H*Ai|| < epsilon: #Follow up 2
                Put AiBj into the maybeInliers set
                n--
                flag = 1
                if flag == 0:
                    break
                elif:
                    iterations ++
                    recalculate H by the new maybeInliers set

    return H
'''    
 