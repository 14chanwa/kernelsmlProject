#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:41:59 2018

@author: imke_mayer
"""

from kernelmlProject import *

import numpy as np

#%%
# Read training set 1
Xtr0 = np.genfromtxt('./data/Xtr0_mat50.csv', delimiter=' ')

# Read training labels 1
Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
# Discard first line
Ytr0 = Ytr0[1:]
# Get only labels
Ytr0_labels = Ytr0[:, 1]
# Map the 0/1 labels to -1/1
Ytr0_labels = 2*Ytr0_labels-1

# Read test set 1
Xte0 = np.genfromtxt('./data/Xte0_mat50.csv', delimiter=' ')


#%%
# To obtain different training and test sets for cross validation step, include
# permutation in this function or outside?
def split_train_test(X,y,prop=0.9):
    nb_train = int(np.floor(prop*len(X)))
    X_train = X[:nb_train,]
    y_train = y[:nb_train]
    X_test = X[nb_train:,]
    y_test = y[nb_train:]
    return X_train, y_train, X_test, y_test

#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.5) 
  
n = Xtr.shape[0]
lambds = np.logspace(np.log10(0.1), np.log10(100), 10)
acc = np.zeros(len(lambds))
logistic_regression = LogisticRegression(center=False) 
for i in range(len(lambds)):
    # LocisticRegression(center=True) would probably be better but problem with CenteredKernel
    if i == 0:
        logistic_regression.train(Xtr, Ytr, n, lambds[i])
    else:
        logistic_regression.train(Xtr, Ytr, n, lambds[i],logistic_regression.K)

    f = logistic_regression.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with linear kernel logistic regression:", acc[i])
    
    
#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8) 

n = Xtr.shape[0]

# Try different values for the regularizer lambda
# Remark: when using sklearn a good value of C = 1/(2*n*lambda) was C = 0.01
# So we expect to find that lambda ~ 30 is a good value here

lambds = np.logspace(np.log10(0.1), np.log10(100), 10)
acc = np.zeros(len(lambds))

for i in range(len(lambds)):
    svm = SVM(Gaussian_kernel(38.0)) 
    svm.train(Xtr, Ytr, n, lambds[i])

    f = svm.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with gaussian kernel SVM:", acc[i])
    
#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8) 

n = Xtr.shape[0]
# Try different values for the kernel parameter gamma

# First on coarse grid
#gammas = np.logspace(np.log10(10), np.log10(200), 10)

# Second on finer grid
gammas = np.linspace(110, 160, 20)
acc = np.zeros(len(gammas))

for i in range(len(gammas)):
    svm = SVM(Gaussian_kernel(gammas[i])) 
    svm.train(Xtr, Ytr, n, 30)

    f = svm.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with gaussian kernel (gamma = ", gammas[i], ") SVM:",acc[i])
  
#%%

nb_trials = 10
avg_acc = 0.0
for k in range(nb_trials):
    N = Xtr0.shape[0]
    permut = np.random.permutation(N)
    Xtr, Ytr, Xte, Yte = split_train_test(Xtr0[permut,],Ytr0_labels[permut],prop=0.8) 
    n = Xtr.shape[0]

    gamma = 120
    lambd = 30

    svm = SVM(Gaussian_kernel(gamma)) 
    svm.train(Xtr, Ytr, n, 30)
    f = svm.predict(Xte, Xte.shape[0])
    tmp = Yte == np.sign(f)
    acc = np.sum(tmp) / np.size(tmp)
    avg_acc += acc
    print("Accuracy on test with gaussian kernel (gamma = ", gamma, ") SVM:",acc)

print(avg_acc/nb_trials)