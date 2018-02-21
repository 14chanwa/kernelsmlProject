# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:45:50 2018

@author: Quentin
"""

import numpy as np
import kernelsmlProject.algorithms as kmlpa
import kernelsmlProject.kernels as kmlpk

from generate_test_results import generate_submission_file

#%%
# Read training set 0
Xtr0 = np.genfromtxt('./data/Xtr0_mat50.csv', delimiter=' ')
# Read training labels 0
Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
# Discard first line and get only labels
Ytr0 = Ytr0[1:, 1]
# Map the 0/1 labels to -1/1
Ytr0 = 2 * Ytr0 - 1


#%%
# Read training set 1
Xtr1 = np.genfromtxt('./data/Xtr1_mat50.csv', delimiter=' ')
# Read training labels 1
Ytr1 = np.genfromtxt('./data/Ytr1.csv', delimiter=',')
# Discard first line and get only labels
Ytr1 = Ytr1[1:, 1]
# Map the 0/1 labels to -1/1
Ytr1 = 2 * Ytr1 - 1


#%%
# Read training set 2
Xtr2 = np.genfromtxt('./data/Xtr2_mat50.csv', delimiter=' ')
# Read training labels 2
Ytr2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')
# Discard first line and get only labels
Ytr2 = Ytr2[1:, 1]
# Map the 0/1 labels to -1/1
Ytr2 = 2 * Ytr2 - 1


#%%
print(">>> Set 0")


gamma = 120
lambd = 30

svm0 = kmlpa.SVM(kmlpk.Gaussian_kernel(gamma), center=True) 
svm0.train(Xtr0, Ytr0, Xtr0.shape[0], lambd)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%
print(">>> Set 1")


gamma = 5
lambd = 30

svm1 = kmlpa.SVM(kmlpk.Gaussian_kernel(gamma), center=True) 
svm1.train(Xtr1, Ytr1, Xtr1.shape[0], lambd)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%
print(">>> Set 2")


gamma = 100
lambd = 50

svm2 = kmlpa.SVM(kmlpk.Gaussian_kernel(gamma), center=True) 
svm2.train(Xtr2, Ytr2, Xtr2.shape[0], lambd)
# Training accuracy
f = svm2.get_training_results()
tmp = Ytr2 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%

generate_submission_file(svm0, svm1, svm2, use_bow=True)

