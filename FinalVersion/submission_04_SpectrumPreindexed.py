#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:47:08 2018

@author: imke_mayer
"""


import numpy as np
import kernelsmlProject.algorithms as kmlpa
import kernelsmlProject.kernels as kmlpk

from generate_test_results import generate_submission_file_2


#%%
# Read training set 0
Xtr0 = np.loadtxt('./data/Xtr0.csv', dtype = bytes, delimiter="\n").astype(str)
# Read training labels 0
Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
# Discard first line and get only labels
Ytr0 = Ytr0[1:,1]
# Map the 0/1 labels to -1/1
Ytr0 = 2*Ytr0-1



#%%
# Read training set 1
Xtr1 = np.loadtxt('./data/Xtr1.csv', dtype = bytes, delimiter="\n").astype(str)
# Read training labels 1
Ytr1 = np.genfromtxt('./data/Ytr1.csv', delimiter=',')
# Discard first line and get only labels
Ytr1 = Ytr1[1:, 1]
# Map the 0/1 labels to -1/1
Ytr1 = 2 * Ytr1 - 1


#%%
#~ # Read training set 2
#~ Xtr2 = np.loadtxt('./data/Xtr2.csv', dtype = bytes, delimiter="\n").astype(str)
#~ # Read training labels 2
#~ Ytr2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')
#~ # Discard first line and get only labels
#~ Ytr2 = Ytr2[1:, 1]
#~ # Map the 0/1 labels to -1/1
#~ Ytr2 = 2 * Ytr2 - 1

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


k = 5
lambd = 0.028147977941176464
print("k=", k, "lambda=", lambd)

n = Xtr0.shape[0]

current_kernel = kmlpk.SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)
svm0 = kmlpa.SVM(current_kernel, center=True) 
svm0.train(Xtr0, Ytr0, n, lambd)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 0.864375) # expected perf 0.8


#%%
#~ print(">>> Set 1")


#~ k = 5
#~ lambd = 0.07174124513618677

#~ n = Xtr1.shape[0]

#~ current_kernel = kmlpk.SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)
#~ svm1 = kmlpa.SVM(current_kernel, center=True) 
#~ svm1.train(Xtr1, Ytr1, n, lambd)
#~ # Training accuracy
#~ f = svm1.get_training_results()
#~ tmp = Ytr1 == np.sign(f)
#~ accuracy = np.sum(tmp) / np.size(tmp)
#~ print("Training accuracy:", accuracy, "expected~", 0.931875) # expected perf 0.86

#~ print(">>> Set 1")


#~ k = 6
#~ lambd = 0.040521978021978024

#~ n = Xtr1.shape[0]

#~ current_kernel = kmlpk.SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)
#~ svm1 = kmlpa.SVM(current_kernel, center=True) 
#~ svm1.train(Xtr1, Ytr1, n, lambd)
#~ # Training accuracy
#~ f = svm1.get_training_results()
#~ tmp = Ytr1 == np.sign(f)
#~ accuracy = np.sum(tmp) / np.size(tmp)
#~ print("Training accuracy:", accuracy, "expected~", 0.974375) # expected perf 0.8725


print(">>> Set 1")


k = 7
lambd = 0.02451795212765957

n = Xtr1.shape[0]

current_kernel = kmlpk.SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)
svm1 = kmlpa.SVM(current_kernel, center=True) 
svm1.train(Xtr1, Ytr1, n, lambd)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 0.99875) # expected perf 0.8875


#%%
#~ print(">>> Set 2")


#~ k = 4
#~ lambd = 0.24517952127659579
#~ n = Xtr2.shape[0]

#~ current_kernel = kmlpk.SpectrumKernelPreindexed(k,lexicon={"A":0, "T":1, "C":2, "G":3},enable_joblib=True)
#~ svm2 = kmlpa.SVM(current_kernel, center=True) 
#~ svm2.train(Xtr2, Ytr2, n, lambd)
#~ # Training accuracy
#~ f = svm2.get_training_results()
#~ tmp = Ytr2 == np.sign(f)
#~ accuracy = np.sum(tmp) / np.size(tmp)
#~ print("Training accuracy:", accuracy, "expected~", 0.700625) # expected perf 0.65


#%%
print(">>> Set 2")


gamma = 10
lambd = 0.0002


svm2 = kmlpa.SVM(kmlpk.GaussianKernel(gamma), center=True) 
svm2.train(Xtr2, Ytr2, Xtr2.shape[0], lambd)
# Training accuracy
f = svm2.get_training_results()
print(f)
tmp = Ytr2 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%

Xte0 = np.loadtxt('./data/Xte0.csv', dtype = bytes, delimiter="\n").astype(str)
Xte1 = np.loadtxt('./data/Xte1.csv', dtype = bytes, delimiter="\n").astype(str)
Xte2 = np.genfromtxt('./data/Xte2_mat50.csv', delimiter=' ')

generate_submission_file_2(svm0, svm1, svm2, \
    Xte0, len(Xte0), Xte1, len(Xte1), Xte2, Xte2.shape[0])

