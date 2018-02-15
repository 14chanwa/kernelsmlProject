# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:28:24 2018

@author: Quentin
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
# Test sklearn...

from sklearn import svm

clf = svm.SVC()

# Train on all of the training set (should instead perform some kind of cross-
# validation) -> partly resolved, need to do more cross-checks?
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels) 

clf.fit(Xtr, Ytr)

# Accuracy on training set
Yte_results = clf.predict(Xte)
tmp = Yte == Yte_results
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy with linear SVM:", accuracy)
