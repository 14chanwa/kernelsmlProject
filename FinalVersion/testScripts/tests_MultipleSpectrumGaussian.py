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

f = open('workfile', 'w')
f.write('test\n')

#~ #%%
#~ # Xtr0$

#~ f.write('Xtr0\n')

#~ list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ n = Xtr0.shape[0]
#~ results={}

#~ for gamma in np.arange(4, 10, 0.3):
        
        #~ print("gamma=", gamma)
        #~ print("Xtr0 -- list_k=", list_k)

        #~ current_kernel = MultipleSpectrumGaussianKernel(
                #~ list_k=list_k,
                #~ lexicon={"A":0, "T":1, "C":2, "G":3},
                #~ gamma=gamma
                #~ )
        #~ svm = SVM(current_kernel, center=True)

        #~ res = k_fold_cross_validation(
                #~ Xtr=Xtr0, 
                #~ Ytr=Ytr0_labels, 
                #~ n=len(Xtr0), 
                #~ kernel=current_kernel, 
                #~ algorithm=svm, 
                #~ k=5,
                #~ lambd_min=1e-7,#1e-4,#0.12971666666666665, #1e-4,#0.083425,#1e-4,
                #~ lambd_max=1e-4,#1e0,#0.16674999999999998, #1e0,#0.138975,#1e0, 
                #~ steps=6, 
                #~ depth=8
                #~ )
        #~ print(res)
        #~ results[gamma] = res
        #~ f.write(str(gamma) + ", " + str(res) + "\n")

#~ for r in results.keys():
        #~ print(r, results[r])

# gamma=10
# Final bounds: [ 9.050925925925925e-06 , 9.108024691358024e-06 ] with test accuracy in [ 0.759 , 0.759 ]
# Final bounds: [ 7.748971193415637e-06 , 7.75034293552812e-06 ] with test accuracy in [ 0.7505 , 0.7505 ]
# Final bounds: [ 9.749885688157293e-06 , 9.750038103947569e-06 ] with test accuracy in [ 0.7545 , 0.7545 ]
# Final bounds: [ 9.149977137631459e-06 , 9.150007620789515e-06 ] with test accuracy in [ 0.751 , 0.751 ]

# gamma=5
# Final bounds: [ 1.1972908093278467e-05 , 1.199531321444902e-05 ] with test accuracy in [ 0.756 , 0.756 ]

# gamma=7
# Final bounds: [ 6.517261088248744e-06 , 6.520995275110502e-06 ] with test accuracy in [ 0.7590000000000001 , 0.7590000000000001 ]
# tr_acc ~ 1.0

# gamma = 1
# Final bounds: [ 4.043362292333486e-06 , 4.073235787227557e-06 ] with test accuracy in [ 0.7525000000000001 , 0.752 ]


#%%
# Xtr1

#~ f.write('Xtr1\n')

#~ list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ n = Xtr1.shape[0]
#~ results={}

#~ for gamma in np.arange(0.1, 10, 0.3):

        #~ print("gamma=", gamma)
        #~ print("Xtr1 -- list_k=", list_k)

        #~ current_kernel = MultipleSpectrumGaussianKernel(
                #~ list_k=list_k,
                #~ lexicon={"A":0, "T":1, "C":2, "G":3},
                #~ gamma=gamma
                #~ )
        #~ svm = SVM(current_kernel, center=True)

        #~ res = k_fold_cross_validation(
                        #~ Xtr=Xtr1, 
                        #~ Ytr=Ytr1_labels, 
                        #~ n=len(Xtr1), 
                        #~ kernel=current_kernel, 
                        #~ algorithm=svm, 
                        #~ k=5,
                        #~ lambd_min=1e-7, #1e-4,
                        #~ lambd_max=1e-4, #1e0, 
                        #~ steps=6, 
                        #~ depth=8
                        #~ )
        #~ print(res)
        #~ results[gamma] = res

#~ for r in results.keys():
        #~ f.write(str(r) + ", " + str(results[r]) + "\n")
        #~ print(r, results[r])

# gamma=7
# Final bounds: [ 2.9577732053040696e-05 , 2.958520042676421e-05 ] with test accuracy in [ 0.8895 , 0.8895 ]


#%%
# Xtr2

f.write('Xtr2\n')

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]
results={}

for gamma in np.arange(0.1, 10, 0.3):

        print("gamma=", gamma)
        print("Xtr2 -- list_k=", list_k)

        current_kernel = MultipleSpectrumGaussianKernel(
                list_k=list_k,
                lexicon={"A":0, "T":1, "C":2, "G":3},
                gamma=gamma
                )
        svm = SVM(current_kernel, center=True)

        res = k_fold_cross_validation(
                        Xtr=Xtr2, 
                        Ytr=Ytr2_labels, 
                        n=len(Xtr2), 
                        kernel=current_kernel, 
                        algorithm=svm, 
                        k=5,
                        lambd_min=1e-7, #1e-4,
                        lambd_max=1e-4, #1e0, 
                        steps=6, 
                        depth=8
                        )
        print(res)
        results[gamma] = res

for r in results.keys():
        f.write(str(r) + ", " + str(results[r]) + "\n")
        print(r, results[r])

# gamma=7
# Final bounds: [ 2.7314814814814816e-05 , 3.0037037037037036e-05 ] with test accuracy in [ 0.667 , 0.666 ]
# Final bounds: [ 2.73e-05 , 2.83e-05 ] with test accuracy in [ 0.646 , 0.645 ]
# Final bounds: [ 2.756111111111111e-05 , 2.7735185185185184e-05 ] with test accuracy in [ 0.6519999999999999 , 0.6519999999999999 ]
# Final bounds: [ 2.7e-05 , 2.723148148148148e-05 ] with test accuracy in [ 0.6535 , 0.653 ]

f.close()
