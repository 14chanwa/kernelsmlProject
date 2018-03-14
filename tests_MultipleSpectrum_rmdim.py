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



#~ #%%
#~ # Xtr0

#~ list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ n = Xtr0.shape[0]

#~ print("Xtr0 -- list_k=", list_k)

#~ current_kernel = MultipleSpectrumKernel(
        #~ list_k=list_k,
        #~ lexicon={"A":0, "T":1, "C":2, "G":3},
        #~ remove_dimensions=True
        #~ )
#~ svm = SVM(current_kernel, center=True)

#~ lambd = k_fold_cross_validation(
                #~ Xtr=Xtr0, 
                #~ Ytr=Ytr0_labels, 
                #~ n=len(Xtr0), 
                #~ kernel=current_kernel, 
                #~ algorithm=svm, 
                #~ k=5,
                #~ lambd_min=0.124,#0.12971666666666665, #1e-4,#0.083425,#1e-4,
                #~ lambd_max=0.129,#0.16674999999999998, #1e0,#0.138975,#1e0, 
                #~ steps=6, 
                #~ depth=3
                #~ )

#~ # For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ # Final bounds: [ 0.1398425925925926 , 0.22314814814814815 ] with test accuracy in [ 0.7430000000000001 , 0.7455 ]
#~ # Final bounds: [ 0.13 , 0.1313888888888889 ] with test accuracy in [ 0.7415 , 0.742 ]
#~ # Final bounds: [ 0.12499999999999999 , 0.12833333333333333 ] with test accuracy in [ 0.744 , 0.7444999999999999 ]
#~ # Final bounds: [ 0.124 , 0.12446296296296297 ] with test accuracy in [ 0.7495 , 0.749 ] test_acc ~ 0.97


#~ #%%
#~ # Xtr1

#~ list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ n = Xtr1.shape[0]

#~ print("Xtr1 -- list_k=", list_k)

#~ current_kernel = MultipleSpectrumKernel(
        #~ list_k=list_k,
        #~ lexicon={"A":0, "T":1, "C":2, "G":3},
        #~ remove_dimensions=True
        #~ )
#~ svm = SVM(current_kernel, center=True)

#~ lambd = k_fold_cross_validation(
                #~ Xtr=Xtr1, 
                #~ Ytr=Ytr1_labels, 
                #~ n=len(Xtr1), 
                #~ kernel=current_kernel, 
                #~ algorithm=svm, 
                #~ k=5,
                #~ lambd_min=0.148, #1e-4,
                #~ lambd_max=0.150, #1e0, 
                #~ steps=6, 
                #~ depth=3
                #~ )

#~ # For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ # Final bounds: [ 0.014875 , 0.024125000000000004 ] with test accuracy in [ 0.8860000000000001 , 0.8860000000000001 ]
#~ # Final bounds: [ 0.14833333333333334 , 0.1501851851851852 ] with test accuracy in [ 0.8845000000000001 , 0.885 ]
#~ # Final bounds: [ 0.15305555555555556 , 0.15331481481481482 ] with test accuracy in [ 0.8795 , 0.8795 ]
#~ # Final bounds: [ 0.14944444444444444 , 0.14951851851851852 ] with test accuracy in [ 0.8825 , 0.8825 ] tr_acc ~ 0.986


#%%
# Xtr2

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]

print("Xtr2 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr2, 
                Ytr=Ytr2_labels, 
                n=len(Xtr2), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.2533, #1e-4,
                lambd_max=0.255, #1e0, 
                steps=6, 
                depth=3
                )
# without removing dimensions
# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.05565 , 0.09268333333333333 ] with test accuracy in [ 0.648 , 0.6460000000000001 ]
# Final bounds: [ 0.08239629629629629 , 0.0837679012345679 ] with test accuracy in [ 0.6425 , 0.643 ] tr_acc ~ 0.98

# Final bounds: [ 0.29670370370370364 , 0.3151851851851852 ] with test accuracy in [ 0.6415000000000001 , 0.641 ]
# Final bounds: [ 0.26222222222222225 , 0.27111111111111114 ] with test accuracy in [ 0.6395 , 0.6395 ]
# Final bounds: [ 0.2661111111111112 , 0.267962962962963 ] with test accuracy in [ 0.6419999999999999 , 0.6419999999999999 ]
# Final bounds: [ 0.2533333333333333 , 0.2544444444444445 ] with test accuracy in [ 0.6435000000000001 , 0.644 ]
# Final bounds: [ 0.2545277777777778 , 0.2545907407407408 ] with test accuracy in [ 0.6465 , 0.6465 ] tr_acc ~ 0.88
