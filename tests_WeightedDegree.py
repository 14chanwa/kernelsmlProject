#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 11:10:47 2018

@author: imke_mayer
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 17:29:19 2018

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
Xtr, Ytr, Xte, Yte = split_train_test(Xtr2,Ytr2_labels,prop=0.8)
n = Xtr.shape[0]

Ks = [3,5,8,10,15]
acc = np.zeros(len(Ks))
tr_acc = np.zeros(len(Ks))
lambd = 100 / (2 * n)
for i in range(len(Ks)):
    current_kernel = WDKernel(Ks[i],lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)

    svm = SVM(current_kernel, center=True)

    svm.train(Xtr, Ytr, n, lambd)
    f = svm.predict(Xte, Xte.shape[0])
    
    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("k=", Ks[i], "lambda=", lambd, "Accuracy on test with spectrum kernel SVM:",acc[i])
        
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    tr_acc[i] = np.sum(tmp) / np.size(tmp)
    print("k=", Ks[i], "lambda=", lambd, "Accuracy on training with spectrum kernel SVM::",tr_acc[i])
    

# Tends to overfit even for small values of k (length of k-mers)
# For Xtr2 and k = 3: test accuracy of 55%, training accuracy of 81%
# and for increasing k, training accuracy fast tends to 1 (for k=8, training acc of 96%)
