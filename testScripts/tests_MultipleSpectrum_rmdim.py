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

list_k=[4, 5, 6, 7, 8]
n = Xtr0.shape[0]

print("Xtr0 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr0, 
                Ytr=Ytr0_labels, 
                n=len(Xtr0), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.1,#0.12971666666666665, #1e-4,#0.083425,#1e-4,
                lambd_max=0.2,#0.16674999999999998, #1e0,#0.138975,#1e0, 
                steps=6, 
                depth=3
                )

# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.1398425925925926 , 0.22314814814814815 ] with test accuracy in [ 0.7430000000000001 , 0.7455 ]
# Final bounds: [ 0.13 , 0.1313888888888889 ] with test accuracy in [ 0.7415 , 0.742 ]
# Final bounds: [ 0.12499999999999999 , 0.12833333333333333 ] with test accuracy in [ 0.744 , 0.7444999999999999 ]
# Final bounds: [ 0.124 , 0.12446296296296297 ] with test accuracy in [ 0.7495 , 0.749 ] test_acc ~ 0.97

# For list_k=[1, 2, 3, 4, 5, 6, 7]
# Final bounds: [ 0.1592592592592593 , 0.17407407407407408 ] with test accuracy in [ 0.7445 , 0.744 ]

# For list_k=[2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.15 , 0.15833333333333333 ] with test accuracy in [ 0.748 , 0.748 ]

# For list_k=[3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.17222222222222222 , 0.17592592592592593 ] with test accuracy in [ 0.7565000000000001 , 0.757 ]


# For list_k=[4, 5, 6, 7, 8]
# Final bounds: [ 0.16851851851851854 , 0.17407407407407408 ] with test accuracy in [ 0.758 , 0.758 ]
# Final bounds: [ 0.1 , 0.1037037037037037 ] with test accuracy in [ 0.7539999999999999 , 0.7535000000000001 ]
# Final bounds: [ 0.15000000000000002 , 0.15833333333333335 ] with test accuracy in [ 0.745 , 0.7445 ]
# Final bounds: [ 0.1661111111111111 , 0.16722222222222222 ] with test accuracy in [ 0.754 , 0.754 ]


# For list_k=[3, 5, 6, 7, 8]
# Final bounds: [ 0.10185185185185186 , 0.10462962962962963 ] with test accuracy in [ 0.739 , 0.739 ]
# Final bounds: [ 0.08888888888888889 , 0.09444444444444444 ] with test accuracy in [ 0.7465 , 0.7464999999999999 ]

# For list_k=[4, 6, 7, 8]
# Final bounds: [ 0.16851851851851854 , 0.17222222222222222 ] with test accuracy in [ 0.7484999999999999 , 0.749 ]

# For list_k=[4, 5, 7, 8]
# Final bounds: [ 0.1 , 0.10925925925925926 ] with test accuracy in [ 0.7455 , 0.7444999999999999 ]

# For list_k=[4, 5, 6, 8]
# Final bounds: [ 0.10277777777777779 , 0.10694444444444445 ] with test accuracy in [ 0.7515000000000001 , 0.752 ]
# Final bounds: [ 0.1138888888888889 , 0.11944444444444444 ] with test accuracy in [ 0.7455 , 0.7450000000000001 ]

# For list_k=[4, 5, 6, 7]
# Final bounds: [ 0.11851851851851852 , 0.12407407407407407 ] with test accuracy in [ 0.7505000000000001 , 0.75 ]

# list_k=[1, 4, 5, 6, 7, 8]
# Final bounds: [ 0.11111111111111112 , 0.11666666666666667 ] with test accuracy in [ 0.741 , 0.741 ]



#%%
# Xtr1

list_k=[5, 6, 7]
n = Xtr1.shape[0]

print("Xtr1 -- list_k=", list_k)

current_kernel = MultipleSpectrumKernel(
        list_k=list_k,
        lexicon={"A":0, "T":1, "C":2, "G":3},
        remove_dimensions=True
        )
svm = SVM(current_kernel, center=True)

lambd = k_fold_cross_validation(
                Xtr=Xtr1, 
                Ytr=Ytr1_labels, 
                n=len(Xtr1), 
                kernel=current_kernel, 
                algorithm=svm, 
                k=5,
                lambd_min=0.09, #1e-4,
                lambd_max=0.13, #1e0, 
                steps=6, 
                depth=3
                )

# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.014875 , 0.024125000000000004 ] with test accuracy in [ 0.8860000000000001 , 0.8860000000000001 ]
# Final bounds: [ 0.14833333333333334 , 0.1501851851851852 ] with test accuracy in [ 0.8845000000000001 , 0.885 ]
# Final bounds: [ 0.15305555555555556 , 0.15331481481481482 ] with test accuracy in [ 0.8795 , 0.8795 ]
# Final bounds: [ 0.14944444444444444 , 0.14951851851851852 ] with test accuracy in [ 0.8825 , 0.8825 ] tr_acc ~ 0.986

# For list_k=[1, 2, 3, 4, 5, 6, 7]
# Final bounds: [ 0.1703703703703704 , 0.17407407407407408 ] with test accuracy in [ 0.882 , 0.8815 ]

# For list_k=[2, 3, 4, 5, 6, 7]
# Final bounds: [ 0.12222222222222222 , 0.1259259259259259 ] with test accuracy in [ 0.884 , 0.8844999999999998 ]

# For list_k=[3, 4, 5, 6, 7]
# Final bounds: [ 0.14166666666666666 , 0.1527777777777778 ] with test accuracy in [ 0.883 , 0.883 ]

# For list_k=[4, 5, 6, 7]
# Final bounds: [ 0.12222222222222222 , 0.1259259259259259 ] with test accuracy in [ 0.885 , 0.8855000000000001 ]

# For list_k=[4, 5, 6, 7, 8]
# Final bounds: [ 0.17222222222222222 , 0.17592592592592593 ] with test accuracy in [ 0.8845000000000001 , 0.8845000000000001 ]

# For list_k=[5, 6, 7]
# Final bounds: [ 0.10555555555555557 , 0.10925925925925926 ] with test accuracy in [ 0.885 , 0.8855000000000001 ]
# Final bounds: [ 0.12222222222222222 , 0.1259259259259259 ] with test accuracy in [ 0.8870000000000001 , 0.8870000000000001 ]
# Final bounds: [ 0.12092592592592592 , 0.12231481481481482 ] with test accuracy in [ 0.8789999999999999 , 0.8789999999999999 ]
# Final bounds: [ 0.11 , 0.11222222222222222 ] with test accuracy in [ 0.8879999999999999 , 0.8879999999999999 ]

# For list_k=[6, 7]
# Final bounds: [ 0.1 , 0.10092592592592593 ] with test accuracy in [ 0.8799999999999999 , 0.8799999999999999 ]
# Final bounds: [ 0.05277777777777778 , 0.05694444444444444 ] with test accuracy in [ 0.881 , 0.881 ]
# Final bounds: [ 0.04611111111111111 , 0.04796296296296296 ] with test accuracy in [ 0.8855000000000001 , 0.885 ]

# For list_k=[5, 7]
# Final bounds: [ 0.08055555555555556 , 0.08611111111111111 ] with test accuracy in [ 0.885 , 0.8845000000000001 ]

# For list_k=[5]
# Final bounds: [ 0.08611111111111112 , 0.09722222222222222 ] with test accuracy in [ 0.8705 , 0.8714999999999999 ]

# For list_k=[7]
# Final bounds: [ 0.05 , 0.05462962962962963 ] with test accuracy in [ 0.874 , 0.873 ]


#%%
# Xtr2

list_k=[3]
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
                lambd_min=0.1, #1e-4,
                lambd_max=0.2, #1e0, 
                steps=6, 
                depth=3
                )

# For list_k=[1, 2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.29670370370370364 , 0.3151851851851852 ] with test accuracy in [ 0.6415000000000001 , 0.641 ]
# Final bounds: [ 0.26222222222222225 , 0.27111111111111114 ] with test accuracy in [ 0.6395 , 0.6395 ]
# Final bounds: [ 0.2661111111111112 , 0.267962962962963 ] with test accuracy in [ 0.6419999999999999 , 0.6419999999999999 ]
# Final bounds: [ 0.2533333333333333 , 0.2544444444444445 ] with test accuracy in [ 0.6435000000000001 , 0.644 ]
# Final bounds: [ 0.2545277777777778 , 0.2545907407407408 ] with test accuracy in [ 0.6465 , 0.6465 ] tr_acc ~ 0.88

# For list_k=[2, 3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.27222222222222225 , 0.2759259259259259 ] with test accuracy in [ 0.6409999999999999 , 0.6409999999999999 ]

# For list_k=[3, 4, 5, 6, 7, 8]
# Final bounds: [ 0.23472222222222222 , 0.24861111111111112 ] with test accuracy in [ 0.633 , 0.6325000000000001 ]

# For list_k=[4, 5, 6, 7, 8]
# Final bounds: [ 0.24166666666666667 , 0.25 ] with test accuracy in [ 0.6384999999999998 , 0.6385 ]
# tracc 0.9

# For list_k=[5, 6, 7, 8]
# Final bounds: [ 0.2 , 0.20277777777777778 ] with test accuracy in [ 0.6405000000000001 , 0.6405000000000001 ]
# tracc 0.93

# For list_k=[6, 7, 8]
# Final bounds: [ 0.2518518518518519 , 0.2592592592592593 ] with test accuracy in [ 0.614 , 0.6139999999999999 ]
# tracc 0.93

# For list_k=[5, 6, 7]
# Final bounds: [ 0.2 , 0.20231481481481484 ] with test accuracy in [ 0.6405000000000001 , 0.641 ]
# tracc 0.89
# Final bounds: [ 0.15648148148148147 , 0.16944444444444445 ] with test accuracy in [ 0.6385 , 0.6395000000000001 ]

# For list_k=[4, 5, 6, 7]
# Final bounds: [ 0.2277777777777778 , 0.2351851851851852 ] with test accuracy in [ 0.6445000000000001 , 0.6455 ]
# tracc 0.85

# For list_k=[4, 5, 6]
# Final bounds: [ 0.23703703703703705 , 0.24074074074074073 ] with test accuracy in [ 0.643 , 0.643 ]
# tracc 0.8

# For list_k=[4, 5]
# Final bounds: [ 0.20740740740740743 , 0.2101851851851852 ] with test accuracy in [ 0.6435000000000001 , 0.643 ]
# Final bounds: [ 0.15 , 0.15925925925925927 ] with test accuracy in [ 0.6455 , 0.6439999999999999 ]
# Final bounds: [ 0.1 , 0.10694444444444445 ] with test accuracy in [ 0.6519999999999999 , 0.6515000000000001 ]
# tracc 0.75

# For list_k=[4]
# Final bounds: [ 0.16296296296296298 , 0.1703703703703704 ] with test accuracy in [ 0.6455 , 0.6455 ]
# Final bounds: [ 0.16703703703703704 , 0.16777777777777775 ] with test accuracy in [ 0.642 , 0.642 ]
# tracc 0.7

# For list_k=[5]
# Final bounds: [ 0.10185185185185186 , 0.10925925925925926 ] with test accuracy in [ 0.6300000000000001 , 0.6315000000000001 ]
# tracc 0.79

# For list_k=[4, 6]
# Final bounds: [ 0.1277777777777778 , 0.1416666666666667 ] with test accuracy in [ 0.6420000000000001 , 0.6435000000000001 ]
# tracc 0.79

# For list_k=[6]
# Final bounds: [ 0.1 , 0.10694444444444445 ] with test accuracy in [ 0.6265 , 0.6285000000000001 ]
# tracc 0.87

# For list_k=[3]
# Final bounds: [ 0.1 , 0.10462962962962963 ] with test accuracy in [ 0.6285000000000001 , 0.6275000000000001 ]
# tracc 0.65
