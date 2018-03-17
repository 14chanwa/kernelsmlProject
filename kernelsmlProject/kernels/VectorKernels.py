# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:56:16 2018


Implements kernels and associated functions. A kernel instance should be an
object with methods:
    evaluate(self, x, y): get kernel value for points (x, y).
    compute_K_train(self, Xtr, n): get training kernel Gram matrix.
    compute_K_test(self, Xtr, n, Xte, m, verbose=True): get train
        vs test kernel Gram matrix.


Multithreading capabilities:
    If the types of the object enables it (e.g. gaussian or laplacian
    kernels), use numba to jit and parallelize the operations.
    Otherwise, use joblib to multithread.


@author: Quentin & imke_mayer
"""

import numpy as np
import numba
import time
from kernelsmlProject.kernels.AbstractKernels import *


########################################################################
### LinearKernel                                                         
########################################################################


class LinearKernel(Kernel):
    """
        LinearKernel
    """

    def __init__(self, enable_joblib=False):
        super().__init__(enable_joblib)

    def evaluate(self, x, y):
        """
            LinearKernel.evaluate
            Compute x.dot(y)

            Parameters
            ----------
            x: np.array.
            y: np.array.

            Returns
            ----------
            res: float.
        """

        return x.dot(y.transpose())
    
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            LinearKernel.compute_K_train
            Compute K from data.

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K: np.array.
        """

        if verbose:
            print("LinearKernel.compute_K_train")
        
        Ktr = Xtr.dot(Xtr.T)
        
        if verbose:
            print("end")
        
        return Ktr
    
    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            LinearKernel.compute_K_test
            Gets the matrix K_t = [K(t_i, x_j)] where t_i is the ith test sample
            and x_j is the jth training sample.

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            Xte: list(object).
                Test data.
            m: int.
                Length of Xte.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K_te: np.array (shape=(m,n)).
        """

        if verbose:
            print("LinearKernel.compute_K_test")

        K_te = Xte.dot(Xtr.T)

        if verbose:
            print("end")

        return K_te


########################################################################
### PolyKernel                                                         
########################################################################


class PolyKernel(Kernel):
    """
        PolyKernel
    """

    def __init__(self, d=2, c=0, enable_joblib=False):
        super().__init__(enable_joblib)
        self.d = d
        self.c = c

    def evaluate(self, x, y):
        """
            PolyKernel.evaluate
            Compute x.dot(y)

            Parameters
            ----------
            x: np.array.
            y: np.array.

            Returns
            ----------
            res: float.
        """

        return np.power(x.dot(y.transpose()) + self.c, self.d)
    
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            PolyKernel.compute_K_train
            Compute K from data.

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K: np.array.
        """

        if verbose:
            print("PolyKernel.compute_K_train")
        
        Ktr = Xtr.dot(Xtr.T)
        
        Ktr += self.c
        Ktr = np.power(Ktr, self.d)
        
        if verbose:
            print("end")
        
        return Ktr
    
    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            PolyKernel.compute_K_test
            Gets the matrix K_t = [K(t_i, x_j)] where t_i is the ith test sample
            and x_j is the jth training sample.

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            Xte: list(object).
                Test data.
            m: int.
                Length of Xte.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K_te: np.array (shape=(m,n)).
        """

        if verbose:
            print("PolyKernel.compute_K_test")

        K_te = Xte.dot(Xtr.T)
        
        K_te += self.c
        K_te = np.power(K_te, self.d)

        if verbose:
            print("end")

        return K_te
    

########################################################################
### GaussianKernel                                                         
########################################################################


@numba.jit(nopython=True, nogil=True, cache=True)
def _jit_ev_gaussian(x, y, gamma):
    tmp = x - y
    return np.exp(-gamma * np.dot(tmp, tmp))

@numba.jit(nopython=True, parallel=True, nogil=True)
def _jit_Ktr_gaussian(Xtr, n, gamma):
    K = np.zeros((n, n), dtype=np.float64)

    for i in numba.prange(n):
        for j in numba.prange(i + 1):
            K[j, i] = _jit_ev_gaussian(Xtr[i], Xtr[j], gamma)

    # Symmetrize
    return K + K.T - np.diag(np.diag(K))

@numba.jit(nopython=True, parallel=True, nogil=True)
def _jit_Kte_gaussian(Xtr, n, Xte, m, gamma):
    K_t = np.zeros((m, n), dtype=np.float64)

    for j in numba.prange(n):
        for k in numba.prange(m):
            K_t[k, j] = _jit_ev_gaussian(Xte[k], Xtr[j], gamma)

    return K_t


class GaussianKernel(Kernel):
    """
        GaussianKernel
    """

    def __init__(self, gamma, enable_joblib=False):
        super().__init__(enable_joblib)
        self.gamma = gamma

    def evaluate(self, x, y):
        """
            GaussianKernel.evaluate
            Compute exp(- gamma * norm(x, y)^2).
            JIT VERSION

            Parameters
            ----------
            x: np.array.
            y: np.array.

            Returns
            ----------
            res: float.
        """

        return _jit_ev_gaussian(x, y, self.gamma)

    @numba.jit(cache=True)
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            GaussianKernel.compute_K_train
            Compute K from data.
            JIT VERSION - OVERRIDES PARENT METHOD

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K: np.array.
        """

        if verbose:
            print("GaussianKernel.compute_K_train")
        
        K_tr = _jit_Ktr_gaussian(Xtr, n, self.gamma)
        
        if verbose:
            print("end")
        
        return K_tr

    @numba.jit(cache=True)
    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            GaussianKernel.compute_K_test
            Gets the matrix K_t = [K(t_i, x_j)] where t_i is the ith test sample
            and x_j is the jth training sample.
            NON CENTERED KERNEL VALUES version. The centered version is defined
            in derived class CenteredKernel.
            JIT VERSION - OVERRIDES PARENT METHOD

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            Xte: list(object).
                Test data.
            m: int.
                Length of Xte.
            verbose: bool.
                Optional debug output.

            Returns
            ----------
            K_t: np.array (shape=(m,n)).
        """

        if verbose:
            print("GaussianKernel.compute_K_test")

        K_te = _jit_Kte_gaussian(Xtr, n, Xte, m, self.gamma)

        if verbose:
            print("end")

        return K_te
