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
# Test sklearn...

from sklearn import svm

clf = svm.SVC()

# Train on all of the training set (should instead perform some kind of cross-
# validation)
clf.fit(Xtr0, Ytr0_labels)

# Accuracy on training set
Ytr0_results = clf.predict(Xtr0)
tmp = Ytr0_labels == Ytr0_results
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy with linear SVM:", accuracy)
