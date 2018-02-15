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
<<<<<<< HEAD
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
=======
# Test sklearn with some linear kernel using provided bag-of-words features
>>>>>>> 68ece8d265efa90b8371393b625eae20482b4442

from sklearn import svm

clf = svm.LinearSVC()

<<<<<<< HEAD
# Train on all of the training set (should instead perform some kind of cross-
# validation) -> partly resolved, need to do more cross-checks?
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels) 

clf.fit(Xtr, Ytr)

# Accuracy on training set
Yte_results = clf.predict(Xte)
tmp = Yte == Yte_results
=======
# Train on some part of the training set
# The rest is used for validation
training_proportion = 0.8
N = np.shape(Xtr0)[0]
first_test_index = int(np.floor(0.8 * N))

clf.fit(Xtr0[0:first_test_index], Ytr0_labels[0:first_test_index])

# Accuracy on training set
Ytr0_results = clf.predict(Xtr0[first_test_index:N])
tmp = Ytr0_labels[first_test_index:N] == Ytr0_results
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy on test with linear SVM:", accuracy) # 0.6125


#%%
# Test sklearn with some gaussian rbf kernel using provided bag-of-words 
# features

from sklearn import svm

clf = svm.SVC(kernel='rbf', gamma=38, C=0.1)

# Train on some part of the training set
# The rest is used for validation
training_proportion = 0.8
N = np.shape(Xtr0)[0]
first_test_index = int(np.floor(0.8 * N))

clf.fit(Xtr0[0:first_test_index], Ytr0_labels[0:first_test_index])

# Accuracy on training set
Ytr0_results = clf.predict(Xtr0[first_test_index:N])
tmp = Ytr0_labels[first_test_index:N] == Ytr0_results
>>>>>>> 68ece8d265efa90b8371393b625eae20482b4442
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy on test set with gaussian rbf SVM:", accuracy) # 0.6225
