# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 16:26:21 2018

@author: shashidhar
"""
# Linear regression using gradient descent from scratch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import data and read
filename=("C:/Users/shash/Downloads/housing.csv")
names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
dataset = pd.read_csv(filename,delim_whitespace=True,names=names)

# Rows and coloumns
print(dataset.shape)

# Checking corelation
pd.set_option('precision', 2)
dataset.corr(method='pearson')

#Drop unwanted
dataset = dataset.drop(['CRIM','ZN','INDUS','NOX','AGE','DIS','RAD'], axis = 1)
dataset.head()

#Split dataset into train and test
array = dataset.values
Y = array[:,6]
X = array[:,0:6]
X = (X - X.mean()) / X.std()
X
CostL = []
CostT = []
n = len(Y)
ones = np.ones((n,1))
X = np.hstack((ones, X))
validation_size = 0.20
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)
X_T, X_V, Y_T, Y_V = train_test_split(X_train, Y_train, test_size=validation_size, random_state=seed)
print (X_T.shape, Y_T.shape)
print (X_V.shape, Y_V.shape)
alpha = 0.01 #Step size
iterations = 2000 #No. of iterations
np.random.seed(123) #Set the seed
theta = np.random.rand(7) #Pick some random values to start with

def Gradient(X_T,theta,Alpha,Y_T,iterations,X_V,Y_V):
    PrCost = 0
    m = len(Y_V)
    q = len(Y_T)
    X_Transposed = X_T.transpose()
    for i in range(iterations):
        Diff = np.dot(X_V,theta)-Y_V
        Cost = np.sum(Diff**2)/(2*m)
        Diff1 = np.dot(X_T,theta)-Y_T
        CostT = np.sum(Diff1**2)/(2*q)
        Gradient = np.dot(X_Transposed,Diff1)/q
        theta = theta - Alpha*Gradient
        CostL.append(Cost)
        CostT.append(CostT)
        if Cost > PrCost and i != 1:
            return theta
        PrCost = Cost
    return theta

def predict(X_Test,Theta):
    return np.dot(X_Test,Theta)
theta = Gradient(X_T,theta,alpha,Y_T,iterations,X_V,Y_V)
type(theta)
a = np.array(theta[np.newaxis])
print(theta)
a.shape
predictions = predict(X_test,theta) 

plt.scatter(Y_test, predictions)    
plt.xlabel("TrueValue")
plt.ylabel("Predictions")    

plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost Validations')
plt.plot(CostL)
plt.show()
CostL    
plt.title('Cost Function J')
plt.xlabel('No. of iterations')
plt.ylabel('Cost Training')
plt.plot(CostL)
plt.show()

# residual sum of squares
ss_res = np.sum((Y_test - predictions) ** 2)

# total sum of squares
ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)

# r-squared
r2 = 1 - (ss_res / ss_tot)

r2