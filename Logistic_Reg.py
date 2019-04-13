# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 17:26:05 2018

@author: shash
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import data and read
Data = pd.read_csv("C:/Users/shash/Desktop/ex2data1.csv", header=None)
X = Data.iloc[:,:-1]
Y = Data.iloc[:,2]
# Print X Y
X
Y
# Vizualization
mask = Y == 1
adm = plt.scatter(X[mask][0].values, X[mask][1].values)
not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

n = len(Y)
ones = np.ones((n,1))
X = np.hstack((ones, X))
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)

alpha = 0.01 #Step size
iterations = 2000 #No. of iterations
np.random.seed(123) #Set the seed
theta = np.random.rand(3) #Pick some random values to start with
theta.shape

def sigmoid(X):
    
    sig = 1/(1+np.exp(-(X))
    return sig

def costfun(X,Y,theta):                                                                                                                                                                              
    cost = -1/m*np.sum(np.multiply(Y,np.log(sigmoid(x@theta)))+
                       np.multiply((1-y),np.log(1-sigmoid(X@theta))))
    return cost

def gradient(X,Y,theta,alpha):
    gradient = ((1/m) * X.T @ (sigmoid(X @ theta) - Y))
    theta = theta - alpha*gradient
    return theta

def train(X_Train,Y_Train,theta,alpha,iterations):
    costL = []
    for i in range(iterations)
    theta = gradient(X_Train,Y_Train,theta,alpha)
    cost = costfun(X_Train,Y_Train,theta)
    costL.append(cost)
    if 1%%500==0:
        print "iter:"+str(i)+"Cost:"+str(cost)
        
    return theta,costL

def predict(X_Test,theta,cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    return predictions

theta,costL= train(X_Train,Y_Train,theta,alpha,iterations)
predicted = predict(theta,X_Test,0.5)
acc = np.mean(predicted == Y_Test)
print(acc * 100)
              