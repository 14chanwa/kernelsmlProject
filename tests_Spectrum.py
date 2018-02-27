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


# Read training set 1
Xtr1 = np.loadtxt('./data/Xtr1.csv', dtype = bytes, delimiter="\n").astype(str)

# Read training labels 1
Ytr1 = np.genfromtxt('./data/Ytr1.csv', delimiter=',')
# Discard first line
Ytr1 = Ytr1[1:]
# Get only labels
Ytr1_labels = Ytr1[:, 1]
# Map the 0/1 labels to -1/1
Ytr1_labels = 2*(Ytr1_labels-0.5)

# Read test set 1
Xte1 = np.genfromtxt('./data/Xte1.csv', dtype = bytes, delimiter="\n").astype(str)


# Read training set 2
Xtr2 = np.loadtxt('./data/Xtr2.csv', dtype = bytes, delimiter="\n").astype(str)

# Read training labels 2
Ytr2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')
# Discard first line
Ytr2 = Ytr2[1:]
# Get only labels
Ytr2_labels = Ytr2[:, 1]
# Map the 0/1 labels to -1/1
Ytr2_labels = 2*(Ytr2_labels-0.5)

# Read test set 2
Xte2 = np.genfromtxt('./data/Xte2.csv', dtype = bytes, delimiter="\n").astype(str)

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

def compute_occurences(Xtr,k):
    n = len(Xtr)
    occs = []
    for i in range(n):
        occs.append({})
        for l in range(len(Xtr[i])-k+1):
            if Xtr[i][l:l+k] in occs[i].keys():
                occs[i][Xtr[i][l:l+k]] += 1
            else:
                occs[i][Xtr[i][l:l+k]] = 1
    return occs      
    
#%%
k = 5
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8)
n = Xtr.shape[0]
Xtr_merged = {}
tries = compute_trie(Xtr,k)
occs = compute_occurences(Xtr,k)
for i in range(len(Xtr)):
    Xtr_merged[i] = (Xtr[i],tries[i],occs[i])
    
Xte_merged = {}
tries=compute_trie(Xte,k)
occs = compute_occurences(Xte,k)
for i in range(len(Xte)):
    Xte_merged[i] = (Xte[i],tries[i],occs[i])

svm = SVM(SpectrumKernel(k))
Cs = np.linspace(0.1,10,10)
acc = np.zeros(len(Cs))
tr_acc = np.zeros(len(Cs))
for i in range(len(Cs)):
    lambd = 1 / (2 * n * Cs[i])
    if i == 0:
        svm.train(Xtr_merged, Ytr, n, lambd)
        KK = svm.K
        f = svm.predict(Xte_merged, Xte.shape[0])
        KK_t = svm.K_t
    else:
        svm.train(Xtr_merged, Ytr, n, lambd, KK)
        f = svm.predict(Xte_merged, Xte.shape[0], KK_t)
    

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with spectrum kernel SVM:",acc[i])
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    tr_acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on training with spectrum kernel SVM::",tr_acc[i])
# C = 1 yields accuracy of 0.63

#%%
# length of k-mers
k = 5

# split training set in training and validation set
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8)
n = Xtr.shape[0]

# pre-process the data by computing retrieval trees and number of occurences of k-mers
Xtr_merged = {}
tries = compute_trie(Xtr,k)
occs = compute_occurences(Xtr,k)
for i in range(len(Xtr)):
    Xtr_merged[i] = (Xtr[i],tries[i],occs[i])
    
Xte_merged = {}
tries=compute_trie(Xte,k)
occs = compute_occurences(Xte,k)
for i in range(len(Xte)):
    Xte_merged[i] = (Xte[i],tries[i],occs[i])

svm = SVM(SpectrumKernel(k))

lambds = np.linspace(1,3,10)
for (i,lambd) in zip(range(len(lambds)),lambds):
    if i == 0:
        svm.train(Xtr_merged, Ytr, n, lambd)
        KK0 = svm.K
        f = svm.predict(Xte_merged, Xte.shape[0])
        KK_t0 = svm.K_t
    else:
        svm.train(Xtr_merged, Ytr, n, lambd, KK0)
        f = svm.predict(Xte_merged, Xte.shape[0], KK_t0)
    

    tmp = Yte == np.sign(f)
    print("Accuracy on test with spectrum kernel SVM (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    print("Accuracy on training with spectrum kernel SVM  (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))
    
# lambda>1 yields good tradeoff between test and training accuracy (0.74 vs. 0.79)


#%%
k = 5
Xtr, Ytr, Xte, Yte = split_train_test(Xtr1,Ytr1_labels,prop=0.8)
n = Xtr.shape[0]
Xtr_merged = {}
tries = compute_trie(Xtr,k)
occs = compute_occurences(Xtr,k)
for i in range(len(Xtr)):
    Xtr_merged[i] = (Xtr[i],tries[i],occs[i])
    
Xte_merged = {}
tries=compute_trie(Xte,k)
occs = compute_occurences(Xte,k)
for i in range(len(Xtr)):
    Xte_merged[i] = (Xte[i],tries[i],occs[i])

svm = SVM(SpectrumKernel(k))

lambds = np.linspace(1,3,10)
for (i,lambd) in zip(range(len(lambds)),lambds):
    if i == 0:
        svm.train(Xtr_merged, Ytr, n, lambd)
        KK1 = svm.K
        f = svm.predict(Xte_merged, Xte.shape[0])
        KK_t1 = svm.K_t
    else:
        svm.train(Xtr_merged, Ytr, n, lambd, KK1)
        f = svm.predict(Xte_merged, Xte.shape[0], KK_t1)
    

    tmp = Yte == np.sign(f)
    print("Accuracy on test with spectrum kernel SVM (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    print("Accuracy on training with spectrum kernel SVM  (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))

# lambd = 2 yields 76% accuracy without overfitting
#%%
k = 6
Xtr, Ytr, Xte, Yte = split_train_test(Xtr2,Ytr2_labels,prop=0.8)
n = Xtr.shape[0]
Xtr_merged = {}
tries = compute_trie(Xtr,k)
occs = compute_occurences(Xtr,k)
for i in range(len(Xtr)):
    Xtr_merged[i] = (Xtr[i],tries[i],occs[i])
    
Xte_merged = {}
tries=compute_trie(Xte,k)
occs = compute_occurences(Xte,k)
for i in range(len(Xtr)):
    Xte_merged[i] = (Xte[i],tries[i],occs[i])

svm = SVM(SpectrumKernel(k))

lambds = np.linspace(10,30,20)
for (i,lambd) in zip(range(len(lambds)),lambds):
    if i == 0:
        svm.train(Xtr_merged, Ytr, n, lambd,KK2)
        KK2 = svm.K
        f = svm.predict(Xte_merged, Xte.shape[0],KK_t2)
        KK_t2 = svm.K_t
    else:
        svm.train(Xtr_merged, Ytr, n, lambd, KK2)
        f = svm.predict(Xte_merged, Xte.shape[0], KK_t2)
    

    tmp = Yte == np.sign(f)
    print("Accuracy on test with spectrum kernel SVM (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    print("Accuracy on training with spectrum kernel SVM  (lambda= ", lambd, "):",np.sum(tmp) / np.size(tmp))

# k = 5 and lambd = 2 yield test acc = 60% and training accuracy = 70% (KK2_5,KK_t2_5)
# k = 4 and lambd = 1 yield test acc = 62% and training accuracy = 64% (KK2_4,KK_t2_4)
# k = 3 and lambd = 1 61 and 62
# k = 6 and lambd = ? 64 and 82
