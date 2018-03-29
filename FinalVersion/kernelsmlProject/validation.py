# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 12:20:30 2018


Implements cross-validation for selection of the parameters.


@author: Quentin & imke_mayer
"""


import numpy as np


def k_fold_cross_validation(Xtr, Ytr, n, kernel, algorithm, k,
        lambd_min=1e-3, lambd_max=1e1, steps=5, depth=2):
    """
    k_fold_cross_validation
    Performs a k-fold cross-validation on the training data given a
    kernel and an algorithm: divides the training data in k buckets,
    use k-1 for training and test on the last one, repeat for all
    combinations of the buckets, and mean the test accuracies.
    Proceed by dichotomy and reduce the search interval.
    
    Parameters
    ----------
    Xtr: list(object).
        Training data.
    Ytr: np.array((n,)).
        Training labels.
    n: int.
        Length of Xtr.
    kernel: Kernel.
        An instance of the kernel to use.
    algorithm: AlgorithmInstance.
        An instance of the algorithm to use, initialized with the
        kernel.
    k: int.
        Number of subsamples used for cross-validation.

    Returns
    ----------
    lambd: float.
        An optimal value of lambda based on cross-validation.
    """
    
    if k < 2 or steps < 3 or depth < 1 or lambd_max <= lambd_min:
        raise Exception("Invalid parameters")

    print("Performing", k, "fold cross-validation")
    # Get indices of the buckets
    div, mod = divmod(n, k)
    indices = np.repeat(np.arange(k), [div+1 if i<mod else div for i in range(k)])
    
    # Shuffle the indices
    np.random.shuffle(indices)
    
    # Compute the full matrix K
    K = kernel.compute_K_train(Xtr, n)
    
    # Compute Ktr, Kte based on the folds
    Ktr = []
    Kte = []
    Ytr_lab = []
    Yte_lab = []
    for i in range(k):
        tmp_tr = K[indices!=i,:]
        Ktr.append(tmp_tr[:,indices!=i])
        tmp_te = K[indices==i,:]
        Kte.append(tmp_te[:,indices!=i])
        Ytr_lab.append(Ytr[indices!=i])
        Yte_lab.append(Ytr[indices==i])

    # For each value of lambda, compute the mean of the test errors
    # on the k runs.
    lambd_low = lambd_min
    lambd_high = lambd_max
    for d in range(depth):
        lambdas_to_test = [lambd_low + (lambd_high - lambd_low) * j / steps for j in range(steps)]
        test_results = []
        
        # Get the mean training acc for a given lambda
        print("Testing lambdas:", lambdas_to_test)
        for j in range(steps):
            te_acc_list = []
            
            for i in range(k):
                algorithm.train(Xtr=None, Ytr=Ytr_lab[i], n=Ktr[i].shape[0], lambd=lambdas_to_test[j], K=Ktr[i])
                
                ftr = algorithm.get_training_results()
                tmp = Ytr_lab[i] == np.sign(ftr)
                tr_acc = np.sum(tmp) / np.size(tmp)
                
                f = algorithm.predict(Xte=None, m=Kte[i].shape[0], K_t=Kte[i])
                tmp = Yte_lab[i] == np.sign(f)
                te_acc = np.sum(tmp) / np.size(tmp)

                print("bucket=", i, "lambd=", lambdas_to_test[j], "train_acc=", tr_acc, "test_acc=", te_acc)
                te_acc_list.append(te_acc)
            # Get mean
            test_results.append(np.mean(te_acc_list))
        
        print("Found test accuracies:", test_results)
        
        # Get the 2 lambda giving the best test accuracies
        best_lambda_indices = np.sort(np.argsort(test_results)[::-1][0:2])
        
        # Get a new reasonably large search interval
        lambd_low = lambdas_to_test[max(best_lambda_indices[0]-1, 0)]
        #~ lambd_low = lambdas_to_test[best_lambda_indices[0]]
        lambd_high = lambdas_to_test[min(best_lambda_indices[1], len(lambdas_to_test)-1)]
        #~ lambd_high = lambdas_to_test[best_lambda_indices[1]]
        
        print("Best test accuracy in: [", test_results[best_lambda_indices[0]], ",", test_results[best_lambda_indices[1]], "]")
        print("New lambda bounds: [", lambd_low, lambd_high, "]")
    print("Final bounds: [", lambd_low, ",", lambd_high, "] with test accuracy in [", test_results[best_lambda_indices[0]], ",", test_results[best_lambda_indices[1]], "]") 

    return lambd_low, lambd_high, test_results[best_lambda_indices[0]], test_results[best_lambda_indices[1]] 
