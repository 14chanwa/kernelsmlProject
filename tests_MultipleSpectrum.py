# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 02:26:40 2018

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
# Xtr0

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr0.shape[0]

print("Xtr0 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3}
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr0, 
                Ytr=Ytr0_labels, 
                n=len(Xtr0), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.12971666666666665, #1e-4,#0.083425,#1e-4,
                lambd_max=0.16674999999999998, #1e0,#0.138975,#1e0, 
                steps=6, 
                depth=3
                )
# For list_k = [4, 5, 6]
# Final bounds: [ 0.11274305555555555 , 0.11891527777777777 ] with test accuracy in [ 0.7470000000000001 , 0.7464999999999999 ]
# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.14308981481481478 , 0.1472046296296296 ] with test accuracy in [ 0.7474999999999999 , 0.7475 ] tr_acc ~ 0.95
# Final bounds: [ 0.15234814814814815 , 0.15440555555555555 ] with test accuracy in [ 0.7530000000000001 , 0.7530000000000001 ] tr_acc ~ 0.95
#%%
# Xtr1

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr1.shape[0]

print("Xtr1 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3}
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr1, 
                Ytr=Ytr1_labels, 
                n=len(Xtr1), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.07416666666666666, #1e-4,
                lambd_max=0.09268333333333333, #1e0, 
                steps=6, 
                depth=3
                )
# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.07416666666666666 , 0.09268333333333333 ] with test accuracy in [ 0.8869999999999999 , 0.8879999999999999 ]
# Final bounds: [ 0.0844537037037037 , 0.0851395061728395 ] with test accuracy in [ 0.8795 , 0.8795 ] tr_acc ~ 0.995


#%%
# Xtr2

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]

print("Xtr2 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3}
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr2, 
                Ytr=Ytr2_labels, 
                n=len(Xtr2), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.05565, #1e-4,
                lambd_max=0.09268333333333333, #1e0, 
                steps=6, 
                depth=3
                )
# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.05565 , 0.09268333333333333 ] with test accuracy in [ 0.648 , 0.6460000000000001 ]
# Final bounds: [ 0.08239629629629629 , 0.0837679012345679 ] with test accuracy in [ 0.6425 , 0.643 ] tr_acc ~ 0.98
# Final bounds: [ 0.07108055555555555 , 0.07313796296296296 ] with test accuracy in [ 0.6485000000000001 , 0.6485000000000001 ]