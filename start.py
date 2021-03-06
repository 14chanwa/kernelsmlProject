# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:47:08 2018

Script corresponding to submission 07 which gives accuracy of 0.78066 on 
the private subset of the test set (0.79333 on public subset):
    
Structure:
    
    1) Load data,
    2) Train three classifiers (one for each set) on Xtrk (k=0,1,2),
    3) Call submission generating function which:
        i) predicts labels for the test sets Xtek (k=0,1,2),
        ii) writes the predicted labels in the file 'Yte.csv'

@author: Quentin & imke_mayer
"""


import numpy as np
from kernelsmlProject import *

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
# Read training set 2
Xtr2 = np.loadtxt('./data/Xtr2.csv', dtype = bytes, delimiter="\n").astype(str)
# Read training labels 2
Ytr2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')
# Discard first line and get only labels
Ytr2 = Ytr2[1:, 1]
# Map the 0/1 labels to -1/1
Ytr2 = 2 * Ytr2 - 1


#%%


print(">>> Set 0")

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
lambd = 0.1242
print("list_k=", list_k, "lambda=", lambd)

n = Xtr0.shape[0]

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm0 = SVM(current_kernel, center=True) 
svm0.train(Xtr0, Ytr0, n, lambd)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 0.97) # expected perf 0.749


#%%


print(">>> Set 1")

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
lambd = 0.14945
print("list_k=", list_k, "lambda=", lambd)

n = Xtr1.shape[0]

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm1 = SVM(current_kernel, center=True) 
svm1.train(Xtr1, Ytr1, n, lambd)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 0.986) # expected perf 0.8825


#%%


print(">>> Set 2")

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
lambd = 0.25455
print("list_k=", list_k, "lambda=", lambd)

n = Xtr2.shape[0]

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm2 = SVM(current_kernel, center=True) 
svm2.train(Xtr2, Ytr2, n, lambd)
# Training accuracy
f = svm2.get_training_results()
tmp = Ytr2 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 0.88) # expected perf 0.6465. Bad!!


#%%


Xte0 = np.loadtxt('./data/Xte0.csv', dtype = bytes, delimiter="\n").astype(str)
Xte1 = np.loadtxt('./data/Xte1.csv', dtype = bytes, delimiter="\n").astype(str)
Xte2 = np.loadtxt('./data/Xte2.csv', dtype = bytes, delimiter="\n").astype(str)

generate_submission_file_2(svm0, svm1, svm2, \
    Xte0, len(Xte0), Xte1, len(Xte1), Xte2, len(Xte2))

