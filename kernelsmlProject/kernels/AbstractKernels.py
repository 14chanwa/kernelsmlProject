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

from abc import ABC, abstractmethod
import numpy as np
import multiprocessing
import joblib as jl
import time


########################################################################
### Kernel                                                         
########################################################################


class Kernel(ABC):

    def __init__(self, enable_joblib=False):
        self.enable_joblib = enable_joblib

    @abstractmethod
    def evaluate(self, x, y):
        pass

    def _fill_train_line(self, Xtr, i):
        res = np.zeros((i + 1,))
        for j in range(i + 1):
            res[j] = self.evaluate(Xtr[i], Xtr[j])
        return res

    def compute_K_train(self, Xtr, n, verbose=True):
        """
            Kernel.compute_K_train
            Compute K from data.
            JOBLIB GENERIC VERSION

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
            print("Called compute_K_train (NON CENTERED VERSION)")
            start = time.time()

        K = np.zeros([n, n], dtype=np.float64)

        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", multiprocessing.cpu_count())

            # Better results when processing from the larger to the shorter
            # line
            results = jl.Parallel(n_jobs=multiprocessing.cpu_count())(
                    jl.delayed(self._fill_train_line)(Xtr, i)
                    for i in range(n - 1, -1, -1)
                )
            for i in range(n):
                K[:i + 1, i] = results[n - 1 - i]
        else:
            for i in range(n):
                K[:i + 1, i] = self._fill_train_line(Xtr, i)

        # Symmetrize
        K = K + K.T - np.diag(K.diagonal())

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K

    def _fill_test_column(self, Xte, Xtr, j, m):
        res = np.zeros((m,))
        for k in range(m):
            res[k] = self.evaluate(Xte[k], Xtr[j])
        return res

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            Kernel.compute_K_test
            Gets the matrix K_t = [K(t_i, x_j)] where t_i is the ith test sample
            and x_j is the jth training sample.
            NON CENTERED KERNEL VALUES version. The centered version is defined
            in derived class CenteredKernel.
            JOBLIB GENERIC VERSION

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
            print("Called compute_K_test (NON CENTERED VERSION)")
            start = time.time()
        K_t = np.zeros((m, n))

        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", multiprocessing.cpu_count())

            results = jl.Parallel(n_jobs=multiprocessing.cpu_count())(
                    jl.delayed(self._fill_test_column)(Xte, Xtr, j, m)
                    for j in range(n)
                )
            for j in range(n):
                K_t[:, j] = results[j]
        else:
            for j in range(n):
                K_t[:, j] = self._fill_test_column(Xte, Xtr, j, m)

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_t


########################################################################
### CenteredKernel                                                         
########################################################################


class CenteredKernel(Kernel):
    """
        CenteredKernel
        Class that implements computing centered kernel values.
    """

    def __init__(self, kernel):
        super().__init__(kernel.enable_joblib)
        self.kernel = kernel

    """
        CenteredKernel.train
        Train the instance to center kernel values.
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data, of the form X = [[x1], ...].
        n: int. 
            Number of lines of Xtr.
        K: np.array(shape=(n, n)). 
            Optional. Kernel Gram matrix if available.    
    """

    def train(self, Xtr, n, K=None, verbose=True):
        self.Xtr = Xtr
        self.n = n
        if K is None:
            # Compute the non-centered Gram matrix
            self.K = self.kernel.compute_K_train(self.Xtr, self.n, verbose=verbose)
        else:
            self.K = K

        self.eta = np.sum(self.K) / np.power(self.n, 2)

        # Store centered kernel
        U = np.ones(self.K.shape) / self.n
        self.centered_K = (np.eye(self.n) - U).dot(self.K) \
            .dot(np.eye(self.n) - U)

    def get_centered_K(self):
        """
            CenteredKernelernel.get_centered_K
            Gets the centered Gram matrix.

            Returns
            ----------
            centered_K: np.array.
        """
        return self.centered_K

    def evaluate(self, x, y):
        """
            CenteredKernel.evaluate
            Compute the centered kernel value of x, y.

            Parameters
            ----------
            x: object.
            y: object.

            Returns
            ----------
            res: float.
                Evaluation of the centered kernel value of x, y.
        """

        tmp = 0.0
        for k in range(self.n):
            tmp += self.kernel.evaluate(x, self.Xtr[k])
            tmp += self.kernel.evaluate(y, self.Xtr[k])
        # print("tmp:", tmp)
        # print("tmp/n:", tmp / self.n)
        # print("centered")
        return self.kernel.evaluate(x, y) - tmp / self.n + self.eta

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            CenteredKernel.compute_K_test
            Compute centered values of the matrix K_t = [K(t_i, x_j)] where t_i is
            the ith test sample and x_j is the jth training sample.
            CENTERED KERNEL VALUES version.
            Overrides Kernel.compute_K_test.

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
            print("Called compute_K_test (CENTERED VERSION)")
            start = time.time()

        # Get the non-centered K test
        K_t_nc = self.kernel.compute_K_test(Xtr, n, Xte, m, verbose)

        # The new K_t is the non-centered matrix
        #   + 1/n * sum_l(K_{k, l}) where k is test index and l is train index
        #   + 1/n * sum_l(K_{l, j}) where l and j are train indices
        #   + 1/n^2 * sum_{l, l'}(K_{l, l'}) where l, l' are train indices
        #   (the latter quantity was stored in self.eta)
        K_t = K_t_nc + (-1 / self.n) * (
                    K_t_nc.dot(np.ones((self.n, self.n))) +
                    np.ones((m, self.n)).dot(self.K))
        K_t += self.eta

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_t
