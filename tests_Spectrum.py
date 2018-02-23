#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 16:26:27 2018

@author: imke_mayer
"""

from kernelsmlProject.kernels import * 
from kernelsmlProject.algorithms import *

import numpy as np
                
        
#%%
# Read training set 0
Xtr0 = np.loadtxt('./data/Xtr0.csv', dtype = bytes, delimiter="\n").astype(str)

# Read training labels 0
Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
# Discard first line
Ytr0 = Ytr0[1:]
# Get only labels
Ytr0_labels = Ytr0[:, 1]
# Map the 0/1 labels to -1/1
Ytr0_labels = 2*(Ytr0_labels-0.5)

# Read test set 0
Xte0 = np.genfromtxt('./data/Xte0.csv', dtype = bytes, delimiter="\n").astype(str)

#%%
def split_train_test(X,y,prop=0.9):
    nb_train = int(np.floor(prop*len(X)))
    X_train = X[:nb_train,]
    y_train = y[:nb_train]
    X_test = X[nb_train:,]
    y_test = y[nb_train:]
    return X_train, y_train, X_test, y_test
    
#%%
def compute_trie(Xtr,k, EOW = '$'):
    n = len(Xtr)
    roots = []
    for i in range(n):
        roots.append({})
        for l in range(len(Xtr[i])-k+1):
            tmp = roots[i]
            for level in range(k):
                tmp = tmp.setdefault(Xtr[i][l+level],{})
            tmp[EOW] = EOW
    return roots      
    
#%% 
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8)
n = Xtr.shape[0]
Xtr_merged = {}
tries = compute_trie(Xtr,3)
for i in range(len(Xtr)):
    Xtr_merged[i] = (Xtr[i],tries[i])
    
Xte_merged = {}
tries=compute_trie(Xte,3)
for i in range(len(Xte)):
    Xte_merged[i] = (Xte[i],tries[i])

svm = SVM(Spectrum_kernel(3))

svm.train(Xtr_merged, Ytr, n, lambd)
 
f = svm.predict(Xte_merged, Xte.shape[0])

tmp = Yte == np.sign(f)
acc[i] = np.sum(tmp) / np.size(tmp)
print("Accuracy on test with gaussian kernel (gamma = ", gammas[i], ") SVM:",acc[i])
