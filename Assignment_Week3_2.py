# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:37:15 2019

@author: PAUL
"""
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt
from pandas import DataFrame
import operator
from functools import reduce

def sigmoid(Y):
    
    return 1.0/(1.0 + np.exp(-Y))
    

def computeLoss(X, Y, theta):
    m = len(Y)
    pred_Y = sigmoid(X * theta)
    
    j_val = (-1.0/m) * (Y.transpose() * np.log(pred_Y) + (1.0 - Y).transpose() * \
             np.log(1.0 - pred_Y))
    return j_val


def gradient(X, Y, theta, alpha):
    m = len(Y)
    pred_Y = sigmoid(X * theta)
    
    theta -= (X.transpose() * (pred_Y - Y)) * (alpha * 1.0/m)  
    return theta
    
    
def gen_sample_data():
     
    # generate 2d classification dataset using sklearn
    X, y = make_blobs(n_samples=100, centers=2, n_features=2) 
                                                                        
    return X, y

    

def train(X, Y, batchSize, lr, max_iter):
    theta = np.mat([[0.0],
                    [0.0],
                    [0.0]]) #a matrix composed of theta0, theta1, theta2
    
    n = len(Y)
    X0 = np.ones(n)
    X = np.c_[X0, X]  #set the x0 to 1 and add to the matrix
    for i in range(max_iter):
        batchIndex = np.random.choice(n, batchSize)
        batchX = X[batchIndex, :] 
        batchY = Y[batchIndex]
        print('theta_0:{0}, theta_1:{1}, theta_2:{2}'.format(theta[0], theta[1],\
              theta[2]))
        print('loss is {0}'.format(computeLoss(batchX, batchY, theta)))
        theta = gradient(batchX, batchY, theta, lr)
    return theta
        
if __name__ == '__main__':
    lr = 0.001
    max_iter = 10000
    batchSize = 80
    X, Y = gen_sample_data()
    Y_copy = Y
    Y_copy = np.asmatrix(Y_copy).transpose()
    theta = train(X, Y_copy, batchSize, lr, max_iter)
    
    # scatter plot, dots colored by class value
    df = DataFrame(dict(x=X[:,0], y=X[:,1], label=Y))
    colors = {0:'red', 1:'blue', 2:'green'}
    fig, ax = plt.subplots()
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    
    # plot the boundary theta0 + theta1*x1 +theta2*x2 = 0
    x1 = np.array(X[:, 1])
    if theta[2] != 0:
        w = np.array(-theta[1]/theta[2])
        b = np.array(-theta[0]/theta[2])
        x2 = w * x1 + b
 
    x1 = x1.reshape(1, -1)
    x1 = reduce(operator.add, x1)
    x2 = reduce(operator.add, x2)
    plt.plot(x1, x2,  color='black')
    
    plt.show()
    
