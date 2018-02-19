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
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8) 

  
n = Xtr.shape[0]
lambds = np.logspace(np.log10(0.1), np.log10(100), 20)
acc = np.zeros(len(lambds))
for i in range(len(lambds)):
    # LocisticRegression(center=True) would probably be better but problem with CenteredKernel
    logistic_regression = LogisticRegression() 
    logistic_regression.train(Xtr, Ytr, n, lambds[i])

    f = logistic_regression.predict(Xte, Xte.shape[0])

    tmp = Yte == (np.sign(f)+1)/2
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with linear kernel logistic regression:", acc[i])
