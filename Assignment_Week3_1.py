# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:37:15 2019

@author: PAUL
"""
import numpy as np



def computeLoss(X, Y, theta):
    m = len(Y)
    pred_Y = X * theta
    loss = (((pred_Y - Y).transpose()) * (pred_Y - Y)) /(2 * m)
    return loss


def gradient(X, Y, theta, alpha):
    m = len(Y)
    theta -= (alpha/m) * (X.transpose() * (X * theta - Y))
    return theta
    
    
def gen_sample_data():
    
    theta = np.random.randint(0, 10, size = (2, 1)) + np.random.random([2, 1])
    theta = np.asmatrix(theta)
    
    m = 100
    X = np.random.randint(0, 100, (m, 1))
    X = np.multiply(X, np.random.random([m, 1]))
    X = np.asmatrix(X)
    X0 = np.ones(m)
    X = np.c_[X0, X]
    
    Y = X * theta
    
    return X, Y, theta

def train(X, Y, batchSize, lr, max_iter):
    theta = np.mat([[0.0],
                   [0.0]])
    
    n = len(Y)
    for i in range(max_iter):
        batchIndex = np.random.choice(n, batchSize)
        batchX = X[batchIndex, :] 
        batchY = Y[batchIndex, :]
        print('w:{0}, b:{1}'.format(theta[1], theta[0]))
        print('loss is {0}'.format(computeLoss(batchX, batchY, theta)))
        theta = gradient(batchX, batchY, theta, lr)
        
if __name__ == '__main__':
    lr = 0.001
    max_iter = 10000
    batchSize = 80
    X, Y, theta = gen_sample_data()
    train(X, Y, batchSize, lr, max_iter)
    print('the rela theta is, w:{0}, b:{1}'.format(theta[1], theta[0]))
    
