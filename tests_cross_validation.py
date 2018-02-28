# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:30:01 2018

@author: Quentin
"""


from kernelsmlProject import *
import numpy as np


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


k=5
current_kernel = SpectrumKernelPreindexed(k, lexicon={"A":0, "T":1, "C":2, "G":3})
svm = SVM(current_kernel, center=True, verbose=False)

lambd = k_fold_cross_validation(
                Xtr=Xtr0, 
                Ytr=Ytr0_labels, 
                n=len(Xtr0), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=1e-4,
                lambd_max=1e0, 
                steps=6, 
                depth=3
                )
# Sample output:
# Final bounds: [ 0.03713333333333333 , 0.09268333333333334 ] with test accuracy in [ 0.736 , 0.735 ]
