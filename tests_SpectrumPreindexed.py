# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 02:26:40 2018

@author: Quentin
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
k = 6
Xtr, Ytr, Xte, Yte = split_train_test(Xtr2,Ytr2_labels,prop=0.8)
n = Xtr.shape[0]

current_kernel = SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},normalize=False,enable_joblib=True)



svm = SVM(current_kernel, center=True)
Cs = np.linspace(0.01,1,60)
acc = np.zeros(len(Cs))
tr_acc = np.zeros(len(Cs))
for i in range(len(Cs)):
    lambd = 100 / (2 * n * Cs[i])
    if i == 0:
        svm.train(Xtr, Ytr, n, lambd)
        KK = svm.K
        f = svm.predict(Xte, Xte.shape[0])
        KK_t = svm.K_t
    else:
        svm.train(Xtr, Ytr, n, lambd, KK)
        f = svm.predict(Xte, Xte.shape[0], KK_t)
    
    
    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("k=", k, "lambda=", lambd, "Accuracy on test with spectrum kernel SVM:",acc[i])
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    tr_acc[i] = np.sum(tmp) / np.size(tmp)
    print("k=", k, "lambda=", lambd, "Accuracy on training with spectrum kernel SVM::",tr_acc[i])
    



#~ #%%
#~ # Read training set 2
#~ Xtr2 = np.genfromtxt('./data/Xtr2_mat50.csv', delimiter=' ')

#~ Xtr, Ytr, Xte, Yte = split_train_test(Xtr2,Ytr2_labels,prop=0.8)

#~ gamma = 10
#~ lambd = 0.0002#1 / (2 * Xtr2.shape[0] * C)


#~ svm2 = SVM(GaussianKernel(gamma), center=True) 
#~ svm2.train(Xtr, Ytr, Xtr.shape[0], lambd)

#~ f = svm2.predict(Xte, Xte.shape[0])
#~ tmp = Yte == np.sign(f)
#~ acc = np.sum(tmp) / np.size(tmp)
#~ print("gamma=", gamma, "lambda=", lambd, lambd, "Accuracy on test:",acc)
    
#~ ftr = svm2.get_training_results()
#~ tmp = Ytr == np.sign(ftr)
#~ tr_acc = np.sum(tmp) / np.size(tmp)
#~ print("gamma=", gamma, "lambda=", lambd, "Accuracy on training:",tr_acc)
