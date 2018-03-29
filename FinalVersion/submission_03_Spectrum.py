#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 16:47:08 2018

@author: imke_mayer
"""


import numpy as np
import kernelsmlProject.algorithms as kmlpa
import kernelsmlProject.kernels as kmlpk

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

def generate_submission_file(classifier0, classifier1, classifier2, use_bow=True):
    if use_bow:
        Xte0 = np.genfromtxt('./data/Xte0_mat50.csv', delimiter=' ')
        m0 = Xte0.shape[0]
        Xte1 = np.genfromtxt('./data/Xte1_mat50.csv', delimiter=' ')
        m1 = Xte1.shape[0]
        Xte2 = np.genfromtxt('./data/Xte2_mat50.csv', delimiter=' ')
        m2 = Xte2.shape[0]
    else:
        Xte0 = np.loadtxt('./data/Xte0.csv', dtype = bytes, delimiter="\n").astype(str)
        m0 = Xte0.shape[0]
        Xte0_merged = {}
        tries=compute_trie(Xte0,k0)
        occs = compute_occurences(Xte0,k0)
        for i in range(len(Xte0)):
            Xte0_merged[i] = (Xte0[i].copy(),tries[i],occs[i])
        Xte0 = Xte0_merged
        Xte1 = np.loadtxt('./data/Xte1.csv', dtype = bytes, delimiter="\n").astype(str)
        m1 = Xte1.shape[0]
        Xte1_merged = {}
        tries=compute_trie(Xte1,k1)
        occs = compute_occurences(Xte1,k1)
        for i in range(len(Xte1)):
            Xte1_merged[i] = (Xte1[i].copy(),tries[i],occs[i])
        Xte1 = Xte1_merged
        Xte2 = np.loadtxt('./data/Xte2.csv', dtype = bytes, delimiter="\n").astype(str)
        m2 = Xte2.shape[0]
        Xte2_merged = {}
        tries=compute_trie(Xte2,k2)
        occs = compute_occurences(Xte2,k2)
        for i in range(len(Xte2)):
            Xte2_merged[i] = (Xte2[i].copy(),tries[i],occs[i])
        Xte2 = Xte2_merged
    
    # Generate labels
    Yte0 = np.array(np.sign(classifier0.predict(Xte0, m0)), dtype=int)
    Yte1 = np.array(np.sign(classifier1.predict(Xte1, m1)), dtype=int)
    Yte2 = np.array(np.sign(classifier2.predict(Xte2, m2)), dtype=int)

    # Map {-1, 1} back to {0, 1}
    Yte0[Yte0 == -1] = 0
    Yte1[Yte1 == -1] = 0
    Yte2[Yte2 == -1] = 0
    
    f = open('results.csv', 'w')
    f.write("Id,Bound\n")
    count = 0
    for i in range(len(Yte0)):
        f.write(str(count)+","+str(Yte0[i])+"\n")
        count += 1
    for i in range(len(Yte1)):
        f.write(str(count)+","+str(Yte1[i])+"\n")
        count += 1
    for i in range(len(Yte2)):
        f.write(str(count)+","+str(Yte2[i])+"\n")
        count += 1
    f.close()

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


k0 = 5
lambd = 2

n = Xtr0.shape[0]

Xtr0_merged = {}
tries=compute_trie(Xtr0,k0)
occs = compute_occurences(Xtr0,k0)
for i in range(len(Xtr0)):
    Xtr0_merged[i] = (Xtr0[i],tries[i],occs[i])
Xtr0 = Xtr0_merged.copy()

svm0 = kmlpa.SVM(kmlpk.SpectrumKernel(k0), center=True) 
svm0.train(Xtr0, Ytr0, n, lambd)
# Training accuracy
f = svm0.get_training_results()
tmp = Ytr0 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%
print(">>> Set 1")


k1 = 5
lambd = 2

n = Xtr1.shape[0]

Xtr1_merged = {}
tries=compute_trie(Xtr1,k1)
occs = compute_occurences(Xtr1,k1)
for i in range(len(Xtr1)):
    Xtr1_merged[i] = (Xtr1[i],tries[i],occs[i])
Xtr1 = Xtr1_merged.copy()

svm1 = kmlpa.SVM(kmlpk.SpectrumKernel(k1), center=True) 
svm1.train(Xtr1, Ytr1, n, lambd)
# Training accuracy
f = svm1.get_training_results()
tmp = Ytr1 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%
print(">>> Set 2")


k2 = 4
lambd = 1

n = Xtr2.shape[0]

Xtr2_merged = {}
tries=compute_trie(Xtr2,k2)
occs = compute_occurences(Xtr2,k2)
for i in range(len(Xtr2)):
    Xtr2_merged[i] = (Xtr2[i],tries[i],occs[i])
Xtr2 = Xtr2_merged.copy()

svm2 = kmlpa.SVM(kmlpk.SpectrumKernel(k2), center=True) 
svm2.train(Xtr2, Ytr2, n, lambd)
# Training accuracy
f = svm2.get_training_results()
tmp = Ytr2 == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy)


#%%

generate_submission_file(svm0, svm1, svm2, use_bow=False)

