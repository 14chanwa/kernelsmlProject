#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 16:41:59 2018

@author: imke_mayer
"""

from kernelsmlProject.kernels import * 
from kernelsmlProject.algorithms import *

import numpy as np

#%%
# Read training set 0
Xtr0 = np.genfromtxt('./data/Xtr0_mat50.csv', delimiter=' ')

# Read training labels 0
Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
# Discard first line
Ytr0 = Ytr0[1:]
# Get only labels
Ytr0_labels = Ytr0[:, 1]
# Map the 0/1 labels to -1/1
Ytr0_labels = 2*(Ytr0_labels-0.5)

# Read test set 0
Xte0 = np.genfromtxt('./data/Xte0_mat50.csv', delimiter=' ')


# Read training set 1
Xtr1 = np.genfromtxt('./data/Xtr1_mat50.csv', delimiter=' ')

# Read training labels 1
Ytr1 = np.genfromtxt('./data/Ytr1.csv', delimiter=',')
# Discard first line
Ytr1 = Ytr1[1:]
# Get only labels
Ytr1_labels = Ytr1[:, 1]
# Map the 0/1 labels to -1/1
Ytr1_labels = 2*(Ytr1_labels-0.5)

# Read test set 1
Xte1 = np.genfromtxt('./data/Xte1_mat50.csv', delimiter=' ')


# Read training set 2
Xtr2 = np.genfromtxt('./data/Xtr2_mat50.csv', delimiter=' ')

# Read training labels 2
Ytr2 = np.genfromtxt('./data/Ytr2.csv', delimiter=',')
# Discard first line
Ytr2 = Ytr2[1:]
# Get only labels
Ytr2_labels = Ytr2[:, 1]
# Map the 0/1 labels to -1/1
Ytr2_labels = 2*(Ytr2_labels-0.5)

# Read test set 1
Xte2 = np.genfromtxt('./data/Xte2_mat50.csv', delimiter=' ')

#%%
# To obtain different training and test sets for cross validation step, include
# permutation in this function or outside?
def split_train_test(X,y,prop=0.9):
    nb_train = int(np.floor(prop*len(X)))
    X_train = X[:nb_train,]
    y_train = y[:nb_train]
    X_test = X[nb_train:,]
    y_test = y[nb_train:]
    return X_train, y_train, X_test, y_test

#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.5) 
  
n = Xtr.shape[0]
lambds = np.logspace(np.log10(0.1), np.log10(100), 10)
acc = np.zeros(len(lambds))
gamma = 120

ridge_regression = RidgeRegression(Gaussian_kernel(gamma), center=True) 
for i in range(len(lambds)):
    if i == 0:
        ridge_regression.train(Xtr, Ytr, n, lambds[i])
    else:
        ridge_regression.train(Xtr, Ytr, n, lambds[i],ridge_regression.K)
    
    f = ridge_regression.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with gaussian kernel ridge regression:", acc[i])
    


#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.5) 
  
n = Xtr.shape[0]
lambds = np.logspace(np.log10(0.1), np.log10(100), 10)
acc = np.zeros(len(lambds))
logistic_regression = LogisticRegression(center=True) 
for i in range(len(lambds)):
    # LocisticRegression(center=True) would probably be better but problem with CenteredKernel
    if i == 0:
        logistic_regression.train(Xtr, Ytr, n, lambds[i])
    else:
        logistic_regression.train(Xtr, Ytr, n, lambds[i],logistic_regression.K)

    f = logistic_regression.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with centered linear kernel logistic regression:", acc[i])


#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.5) 
  
n = Xtr.shape[0]
lambds = np.logspace(np.log10(0.001), np.log10(10), 10)
acc = np.zeros(len(lambds))
logistic_regression = LogisticRegression(Gaussian_kernel(20), center=True) 
for i in range(len(lambds)):
    # LocisticRegression(center=True) would probably be better but problem with CenteredKernel
    if i == 0:
        logistic_regression.train(Xtr, Ytr, n, lambds[i])
    else:
        logistic_regression.train(Xtr, Ytr, n, lambds[i],logistic_regression.K)

    f = logistic_regression.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with centered gaussian kernel logistic regression:", acc[i])
    
#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8) 

n = Xtr.shape[0]

# Try different values for the regularizer lambda
# Remark: when using sklearn a good value of C = 1/(2*n*lambda) was C = 0.01
# So we expect to find that lambda ~ 30 is a good value here

lambds = np.logspace(np.log10(0.1), np.log10(100), 10)
acc = np.zeros(len(lambds))

for i in range(len(lambds)):
    svm = SVM(Gaussian_kernel(38.0), center=True) 
    svm.train(Xtr, Ytr, n, lambds[i])

    f = svm.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with gaussian kernel SVM:", acc[i])
    
#%%
Xtr, Ytr, Xte, Yte = split_train_test(Xtr0,Ytr0_labels,prop=0.8) 

n = Xtr.shape[0]
# Try different values for the kernel parameter gamma

# First on coarse grid
#gammas = np.logspace(np.log10(10), np.log10(200), 10)

# Second on finer grid
gammas = np.linspace(110, 200, 10)
acc = np.zeros(len(gammas))

C = 1
lambd = 1 / (2 * n * C)

for i in range(len(gammas)):
    svm = SVM(Gaussian_kernel(gammas[i]),center=True) 
    svm.train(Xtr, Ytr, n, lambd)

    f = svm.predict(Xte, Xte.shape[0])

    tmp = Yte == np.sign(f)
    acc[i] = np.sum(tmp) / np.size(tmp)
    print("Accuracy on test with gaussian kernel (gamma = ", gammas[i], ") SVM:",acc[i])
  
#%%

nb_trials = 10
avg_acc = 0.0
tr_avg_acc = 0.0
for k in range(nb_trials):
    N = Xtr0.shape[0]
    permut = np.random.permutation(N)
    Xtr, Ytr, Xte, Yte = split_train_test(Xtr0[permut,],Ytr0_labels[permut],prop=0.8) 
    n = Xtr.shape[0]

    gamma = 30 #    (30, 120) for Xtr0; (165-170 or 185, 5 or 160-168) for Xtr1,  (10, ) for Xtr2
    C = 1 #        (1, 0.1) for Xtr0; (1, 5) for Xtr1,    (5, ) for Xtr2
    lambd = 1 / (2 * n * C)

    svm = SVM(Gaussian_kernel(gamma), center=True) 
    svm.train(Xtr, Ytr, n, lambd)
    fte = svm.predict(Xte, Xte.shape[0])
    tmp = Yte == np.sign(fte)
    acc = np.sum(tmp) / np.size(tmp)
    avg_acc += acc
    print("Accuracy on test with gaussian kernel (gamma = ", gamma, ") SVM:",acc)
    
    ftr = svm.get_training_results()
    tmp = Ytr == np.sign(ftr)
    acc = np.sum(tmp) / np.size(tmp)
    tr_avg_acc += acc
    print("Accuracy on training with gaussian kernel (gamma = ", gamma, ") SVM:",acc)

print(avg_acc/nb_trials)
print(tr_avg_acc/nb_trials)

#%%
# C = 1 gamma = 55
# Comparison: same parameters?

N0 = Xtr0.shape[0] * 0.8

C = 0.1
lambd = 1 / (2 * N0 * C)
gamma = 120 # 55 # 120

print("lambd:", lambd)
print("gamma:", gamma)

permut = np.random.permutation(int(N0))
Xtr0_, Ytr0_, Xte0_, Yte0_ = split_train_test(Xtr0[permut,],Ytr0_labels[permut],prop=0.8) 
n = Xtr0_.shape[0]

svm0 = SVM(Gaussian_kernel(gamma), center=True) 
svm0.train(Xtr0_, Ytr0_, n, 30)
f = svm0.predict(Xte0_, Xte0_.shape[0])
tmp = Yte0_ == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)

print("Dataset 0: Accuracy on test with gaussian kernel (gamma = ", gamma, ") SVM:",accuracy)

#
N1 = Xtr1.shape[0] * 0.8

C = 0.1
lambd = 1 / (2 * N1 * C)
gamma = 100 # 55 # 120

print("lambd:", lambd)
print("gamma:", gamma)

permut = np.random.permutation(int(N1))
Xtr1_, Ytr1_, Xte1_, Yte1_ = split_train_test(Xtr1[permut,],Ytr1_labels[permut],prop=0.8) 
n = Xtr1_.shape[0]

svm1 = SVM(Gaussian_kernel(gamma), center=True) 
svm1.train(Xtr1_, Ytr1_, n, 30)
f = svm1.predict(Xte1_, Xte1_.shape[0])
tmp = Yte1_ == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)

print("Dataset 1: Accuracy on test with gaussian kernel (gamma = ", gamma, ") SVM:",accuracy)


#
N2 = Xtr2.shape[0] * 0.8

C = 0.1
lambd = 1 / (2 * N2 * C)
gamma = 120 # 55 # 120

print("lambd:", lambd)
print("gamma:", gamma)

permut = np.random.permutation(int(N2))
Xtr2_, Ytr2_, Xte2_, Yte2_ = split_train_test(Xtr2[permut,],Ytr2_labels[permut],prop=0.8) 
n = Xtr2_.shape[0]

svm2 = SVM(Gaussian_kernel(gamma), center=True) 
svm2.train(Xtr2_, Ytr2_, n, 30)
f = svm2.predict(Xte2_, Xte2_.shape[0])
tmp = Yte2_ == np.sign(f)
accuracy = np.sum(tmp) / np.size(tmp)

print("Dataset 2: Accuracy on test with gaussian kernel (gamma = ", gamma, ") SVM:",accuracy)



print(">>> Generating test results file...")

from generate_test_results import generate_submission_file
generate_submission_file(svm0, svm1, svm2, use_bow=True) # should NOT be written
# like this since test sets 0, 1, 2 are from different datasets

print("end")

#%%

import sklearn

clf = sklearn.svm.SVC(kernel='rbf', gamma=55, C=1)
clf.fit(Xtr, Ytr)
f_skl = clf.predict(Xte)
tmp = Yte == f_skl
accuracy = np.sum(tmp) / np.size(tmp)
print("Accuracy on test set with gaussian rbf SVM:", accuracy) # 0.634375