# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 23:22:30 2018


Implements kernel PCA and visualization functions.


@author: Quentin
"""

from kernelsmlProject.kernels import *
import numpy as np
import scipy.sparse.linalg as sparselinalg
import matplotlib.pyplot as plt
import sklearn.cluster


########################################################################
### K-means                                                         
########################################################################


class KMeans():
    """
        KMeans
        
        Attributes
        ----------
        self.kernel: Kernel.
        self.X_tr: list(object).
        self.n: int.
            Size of the training set.
        self.p: int.
            Number of principal components kept.
        self.K_tr: np.array((n, n)).
            Training Gram matrix.
        self.gamma: np.array((p, n)).
            Computed principal components.
        self.K_te: np.array((m, n)).
            Last used test Gram matrix.
    """
    
    def __init__(self, kernel):
        # if not isinstance(kernel, CenteredKernel):
            # raise Exception("You must provide the KMeans a CenteredKernel")
        self.kernel = kernel
        self._MAX_ITER = 10000
        self._fitted = False
    
    def fit_predict(self, Xtr, n, K, Ktr=None, verbose=True):
        self.X_tr = Xtr
        self.n = n
        self.K_tr = self.kernel.compute_K_train(Xtr, n, verbose=verbose) if Ktr is None else Ktr
        self.K = K
        
        # print(self.K_tr[:10, :10])
        
        # 1. Initialize clusters at linear k-means
        # pi_ = np.zeros((n, K))
        # for i in range(n):
            # pi_[i, i%K] = 1
        # np.random.shuffle(pi_)
        tmp_kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(np.array(Xtr))
        pi_ = np.eye(K)[tmp_kmeans.labels_]
        
        # pi_bis = np.random.normal((n, K))
        pi_bis = np.zeros((n, K))
        t = 0
        
        while not np.array_equal(pi_, pi_bis) and t < self._MAX_ITER:
            t += 1
            # print(t)
            
            pi_bis = np.array(pi_)
            # 2. Update cluster indices
            # theta: theta(i, j) = d(phi(i), m_j)^2, size (n, K)
            
            # Sum of cluster sizes
            clsum = np.sum(pi_, axis=0, keepdims=True)
            clsum[clsum == 0] = 1
            
            theta = np.diag(self.K_tr).reshape((n, 1)).dot(np.ones((1, K)))
            theta -= 2 * self.K_tr.dot(pi_) / clsum
            theta += np.ones((n, 1)).dot(np.sum(np.multiply(pi_, self.K_tr.dot(pi_)), axis=0, keepdims=True)) / np.power(clsum, 2)
            
            # print(theta)
            
            pi_ = np.eye(K)[np.argmin(theta, axis=1)]
            
            # print(pi_)
        
        print(t)
        
        self.X_tr_pi = pi_
        self.X_tr_theta = theta
        self.X_tr_labels = np.argmax(pi_, axis=1)
        self._fitted = True
        
        return self.X_tr_labels
    
    def predict(self, Xte, m, verbose=True):
        if not self._fitted:
            raise Exception("Instance is not fitted")
        
        self.K_te = self.kernel.compute_K_test(self.X_tr, self.n, Xte, m, verbose=verbose)
        
        clsum = np.sum(self.X_tr_pi, axis=0, keepdims=True)
        clsum[clsum == 0] = 1
        
        theta = np.diag(self.K_te).reshape((m, 1)).dot(np.ones((1, self.K)))
        theta -= 2 * self.K_te.dot(self.X_tr_pi) / clsum
        theta += np.ones((m, 1)).dot(np.sum(np.multiply(self.X_tr_pi, self.K_tr.dot(self.X_tr_pi)), axis=0, keepdims=True)) / np.power(clsum, 2)
        
        pi_ = np.eye(self.K)[np.argmin(theta, axis=1)]
        
        self.X_te_pi = pi_
        self.X_te_theta = theta
        self.X_te_labels = np.argmax(pi_, axis=1)
        
        return self.X_te_labels