# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 00:02:40 2018

@author: Quentin
"""

import numpy as np
import kernelsmlProject.algorithms as kmlpa
import kernelsmlProject.kernels as kmlpk


import time


# To obtain different training and test sets for cross validation step, include
# permutation in this function or outside?
def split_train_test(X,y,prop=0.9):
    nb_train = int(np.floor(prop*len(X)))
    X_train = X[:nb_train,]
    y_train = y[:nb_train]
    X_test = X[nb_train:,]
    y_test = y[nb_train:]
    return X_train, y_train, X_test, y_test


# Multithreading with joblib
# On Windows systems, ALL the code that is not definition or import has to be 
# ran inside this loop
if  __name__ == "__main__":
    
    
    enable_joblib = True
    
    
    print(">>> Read data")

    # Read training set 0
    Xtr0 = np.genfromtxt('./data/Xtr0_mat50.csv', delimiter=' ')
    # Read training labels 0
    Ytr0 = np.genfromtxt('./data/Ytr0.csv', delimiter=',')
    # Discard first line and get only labels
    Ytr0 = Ytr0[1:, 1]
    # Map the 0/1 labels to -1/1
    Ytr0 = 2 * Ytr0 - 1


    Xte0 = np.genfromtxt('./data/Xte0_mat50.csv', delimiter=' ')
    
    
    print("\n\n>>> Set 0 train")
    start = time.time()
    
    
    gamma = 120
    lambd = 30
    
    svm0 = kmlpa.SVM(kmlpk.GaussianKernel(gamma, enable_joblib=enable_joblib), center=True) 
    svm0.train(Xtr0, Ytr0, Xtr0.shape[0], lambd)
    # Training accuracy
    f = svm0.get_training_results()
    tmp = Ytr0 == np.sign(f)
    accuracy = np.sum(tmp) / np.size(tmp)
    print("Training accuracy:", accuracy)
    
    
    end = time.time()
    print("Time elapsed:", "{0:.2f}".format(end-start))
    
    
    print("\n\n>>> Set 0 test")
    start = time.time()
    
    
    svm0.predict(Xte0, Xte0.shape[0])
    
    
    end = time.time()
    print("Time elapsed:", "{0:.2f}".format(end-start))
    
    
    N0 = Xtr0.shape[0] * 0.8
    
    
    
    print("\n\n>>> Simple validation test")
    start = time.time()

    C = 0.1
    lambd = 1 / (2 * N0 * C)
    gamma = 120 # 55 # 120
    
    print("lambd:", lambd)
    print("gamma:", gamma)
    
    permut = np.random.permutation(int(N0))
    Xtr0_, Ytr0_, Xte0_, Yte0_ = split_train_test(Xtr0[permut,],Ytr0[permut],prop=0.8) 
    n = Xtr0_.shape[0]
    
    svm0 = kmlpa.SVM(kmlpk.GaussianKernel(gamma, enable_joblib=enable_joblib), center=True) 
    svm0.train(Xtr0_, Ytr0_, n, 30)
    f = svm0.predict(Xte0_, Xte0_.shape[0])
    tmp = Yte0_ == np.sign(f)
    accuracy = np.sum(tmp) / np.size(tmp)
    
    
    print("Test accuracy:", accuracy)
    
    end = time.time()
    print("Time elapsed:", "{0:.2f}".format(end-start))
    
