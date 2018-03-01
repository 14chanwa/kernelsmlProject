# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 23:56:59 2018

@author: Quentin
"""


from kernelsmlProject import * 

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


list_k=[4, 5, 6]#[1, 2, 3, 4, 5, 6, 7, 8]
current_kernel = CenteredKernel(
                MultipleSpectrumKernel(
                list_k=list_k,
                lexicon={"A":0, "T":1, "C":2, "G":3}
                )
        )

visualize_training_data_in_RKHS(current_kernel, Xtr0, Ytr0_labels, len(Xtr0))

#%%
# Read training set 0
Xtr0_bow = np.genfromtxt('./data/Xtr0_mat50.csv', delimiter=' ')

gamma = 5
current_kernel = CenteredKernel(
                GaussianKernel(gamma)
                )

visualize_training_data_in_RKHS(current_kernel, Xtr0_bow, Ytr0_labels, Xtr0_bow.shape[0])
