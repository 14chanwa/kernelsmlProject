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

#~ for _d in np.array([2,3,4,5]):
        
        #~ print("gamma=", gamma)
        #~ print("Xtr0 -- list_k=", list_k)

        #~ current_kernel = MultipleSpectrumPolyKernel(
                #~ list_k=list_k,
                #~ lexicon={"A":0, "T":1, "C":2, "G":3},
                #~ d=_d,
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
        #~ results[_d] = res
        #~ f.write(str(_d) + ", " + str(res) + "\n")

#~ for r in results.keys():
        #~ print(r, results[r])



#%%
# Xtr1

#~ f.write('Xtr1\n')

#~ list_k=[1, 2, 3, 4, 5, 6, 7, 8]
#~ n = Xtr1.shape[0]
#~ results={}

#~ for _d in np.array([2,3,4,5]):

        #~ print("gamma=", gamma)
        #~ print("Xtr1 -- list_k=", list_k)

        #~ current_kernel = MultipleSpectrumPolyKernel(
                #~ list_k=list_k,
                #~ lexicon={"A":0, "T":1, "C":2, "G":3},
                #~ d=_d
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
        #~ results[_d] = res

#~ for r in results.keys():
        #~ f.write(str(r) + ", " + str(results[r]) + "\n")
        #~ print(r, results[r])



#%%
# Xtr2

f.write('Xtr2\n')

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]
results={}

for _d in np.array([2,3,4,5]):

        print("d=", _d)
        print("Xtr2 -- list_k=", list_k)

        current_kernel = MultipleSpectrumPolyKernel(
                list_k=list_k,
                d=_d,
                lexicon={"A":0, "T":1, "C":2, "G":3},
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
        results[_d] = res

for r in results.keys():
        f.write(str(r) + ", " + str(results[r]) + "\n")
        print(r, results[r])


# f.close()


#%%

f.write('Xtr2\n rm_dim\n')

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]
results={}

for _d in np.array([2,3,4,5]):

        print("d=", _d)
        print("Xtr2 -- list_k=", list_k)

        current_kernel = MultipleSpectrumPolyKernel(
                list_k=list_k,
                d=_d,
                lexicon={"A":0, "T":1, "C":2, "G":3},
                remove_dimensions=True, 
                normalize=False
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
        results[_d] = res

for r in results.keys():
        f.write(str(r) + ", " + str(results[r]) + "\n")
        print(r, results[r])

# d = 5
# Final bounds: [ 1e-07 , 2.3333547132201648e-05 ] with test accuracy in [ 0.6435 , 0.6335 ]
# d = 4
# Final bounds: [ 1.7158950617283951e-06 , 1.7197016460905348e-06 ] with test accuracy in [ 0.6685 , 0.6685 ]

f.close()


#%%

list_k=[1, 2, 3, 4, 5, 6, 7, 8]
n = Xtr2.shape[0]
d = 4
lambd = 1.716e-06
print("d=", d)
print("Xtr2 -- list_k=", list_k)
current_kernel = MultipleSpectrumPolyKernel(
                list_k=list_k,
                d=d,
                lexicon={"A":0, "T":1, "C":2, "G":3},
                remove_dimensions=True
                )

svm2 = SVM(current_kernel, center=True)

svm2.train(Xtr2, Ytr2_labels, n, lambd)
# Training accuracy
f = svm2.get_training_results()
tmp = Ytr2_labels == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)
print("Training accuracy:", accuracy, "expected~", 1) # expected perf 0.6685. Bad!!