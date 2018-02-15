#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 18:13:06 2018

@author: imke_mayer
"""

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
def split_train_test(X,y,prop=0.9):
    nb_train = int(np.floor(prop*len(X)))
    X_train = X[:nb_train,]
    y_train = y[:nb_train]
    X_test = X[nb_train:,]
    y_test = y[nb_train:]
    return X_train, y_train, X_test, y_test
    
#%%

def build_kernel(X1, X2, kernel='linear', gamma=None, degree=2):

    n = X1.shape[0]
    m = X2.shape[0]
    if gamma is None:
        gamma = 1/n
        
    K = np.zeros([n,m])
    if kernel=='gaussian':
        if n == m:
            for i in range(n):
                for j in range(i+1):
                    K[i,j] = np.exp(-gamma*np.linalg.norm(X1[i]-X2[j])**2)
                    K[j,i] = K[i,j]
        else:
            for i in range(n):
                for j in range(m):
                    K[i,j] = np.exp(-gamma*np.linalg.norm(X1[i]-X2[j])**2)
    
    elif kernel == 'linear':
        K = X1.dot(X2.T)
    elif kernel == 'polynomial':
        K = X1.dot(X2.T)**degree
    else:
        print("Unknown kernel\n")
    return K
    
def learn_svm(X,y,kernel='linear',gamma=None,degree=2,C=1):
    n = X.shape[0]
    P = matrix(build_kernel(X,X,kernel,gamma,degree))
    q = matrix(y)
    G = matrix(np.append(np.eye(n)*(-y),np.eye(n)*(y),axis=0))
    h = matrix(np.append(np.zeros(n),1./(2*C*n)*np.ones(n),axis=0))

    sol_qp = solvers.qp(P, q, G, h)
    alpha = sol_qp['x']
    return alpha
        
# K = build_kernel(Xtr,Xtr, 'gaussian', gamma = 38)

        
#%%
from cvxopt import matrix, solvers

# Train on some part of the training set
# The rest is used for validation
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels) 

# Test own SVM with some linear kernel using provided bag-of-words features

alpha = learn_svm(Xtr, Ytr)

Kte = build_kernel(Xtr,Xte)
Yte_results = (np.sign(Kte.T.dot(alpha))+1)/2
tmp = Yte == Yte_results
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy on test set with linear SVM:", accuracy)

# Test own SVM with some gaussian rbf kernel using provided bag-of-words 
# features
gamma = 38
C = 0.01
alpha = learn_svm(Xtr,Ytr,kernel='gaussian',gamma=gamma, C=C)

Kte = build_kernel(Xtr,Xte,kernel='gaussian', gamma=gamma)
Yte_results = (np.sign(Kte.T.dot(alpha))+1)/2
tmp = Yte == Yte_results
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy on test set with gaussian rbf SVM:", accuracy)


