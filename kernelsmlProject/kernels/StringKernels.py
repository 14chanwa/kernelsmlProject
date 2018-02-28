# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:56:16 2018


Implements kernels and associated functions. A kernel instance should be an
object with methods:
    evaluate(self, x, y): get kernel value for points (x, y).
    compute_matrix_K(self, Xtr, n): get training kernel Gram matrix.
    compute_K_test(self, Xtr, n, Xte, m, verbose=True): get train
        vs test kernel Gram matrix.


Multithreading capabilities:
    If the types of the object enables it (e.g. gaussian or laplacian
    kernels), use numba to jit and parallelize the operations.
    Otherwise, use joblib to multithread.


@author: Quentin & imke_mayer
"""

import numpy as np
import scipy.sparse as sparse
import multiprocessing
import time
from kernelsmlProject.kernels.AbstractKernels import *


########################################################################
### SpectrumKernel                                                         
########################################################################


class SpectrumKernel(Kernel):
    """
        SpectrumKernel
        STILL NEEDS SOME SPEED UP
    """

    def __init__(self, k, EOW='$', enable_joblib=False):
        super().__init__(enable_joblib)
        self.k = k
        self.EOW = '$'

    def evaluate(self, x, y):
        """
            SpectrumKernel.evaluate
            Compute Phi(x[0]).dot(Phi(y[0]))
            where Phi_u(x[0]) denotes the number of occurences of u in sequence x[0]
            u in {A,T,C,G}^k
            x[0] = sequence
            x[1] = trie
            x[2] = number of occurences of each kmer present in x[0]

            Parameters
            ----------
            x: (string, dictionary, dictionary)
            y: (string, dictionary, dictionary)

            Returns
            ----------
            res: float.
        """

        # print("in evaluate")
        xwords = {}
        count = 0
        for l in range(len(x[0]) - self.k + 1):
            if self.find_kmer(y[1], x[0][l:l + self.k]):
                if x[0][l:l + self.k] in xwords.keys():
                    xwords[x[0][l:l + self.k]] += 1
                else:
                    xwords[x[0][l:l + self.k]] = 1
                    count += 1
        xphi = np.fromiter(xwords.values(), dtype=int)
        yphi = [y[2][key] for key in xwords.keys()]
        # this last part probably takes too long, 
        # maybe precompute the number of occurences outside of the kernel
        # (similarly to the computation of the tries)?
        # for (i,w) in zip(range(len(xwords.keys())),xwords.keys()):
        #    yphi[i] = sum(y[0][j:].startswith(w) for j in range(len(y[0])))

        return xphi.dot(yphi)

    def find_kmer(self, trie, kmer):
        """
            SpectrumKernel.find_kmer
            Finds whether a word kmer is present in a given retrieval tree trie
        """

        tmp = trie
        for l in kmer:
            if l in tmp:
                tmp = tmp[l]
            else:
                return False
        return self.EOW in tmp


########################################################################
### SpectrumKernelPreindexed                                                         
########################################################################


class SpectrumKernelPreindexed(Kernel):
    """
        SpectrumKernel_DNA_preindexed
    """

    def __init__(self, k, lexicon, enable_joblib=False):
        """
            SpectrumKernelPreindexed.__init__

            Parameters
            ----------
            k: int. Length of the substrings to be looked for.
            lexicon: dict. Map key to integer.
        """

        super().__init__(enable_joblib)
        self.k = k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)

        # ~ print(lexicon)
        # ~ print(self.lex_size)

    def get_preindexed_value(self, lookup_str):
        """
            SpectrumKernelPreindexed.get_index

            Preindex lexicon, for instance a -> 1, b -> 2... z -> 26
            Here we work with DNA sequences, thus the lexicon is shallow
            (i.e. size lex_size=4). We take as the computed index simply
            the sum in base lex_size (i.e. for a sequence abcd of length
            k=4 we would have 0 + 1 * 4 + 2 * 4**2 + 3 * 4**3, i.e. an
            indexing on 256 values. We can go reasonably far (indices grow
            with 4^{k-1}, we are limited, I believe, to int32).
            We could concatenate arrays for combining values l < k,
            until some point (the indices grow with (4**k - 1 / 3)).


            Parameters
            ----------
            lookup_str: string.

            Returns
            ----------
            res: int.
        """

        res = 0
        for i in range(self.k):
            res += self.lexicon[lookup_str[i]] * self.lex_size ** i
        # print(lookup_str, res)
        return res

    def evaluate(self, x, y):
        """
            SpectrumKernel.evaluate
            Compute Phi(x[0]).dot(Phi(y[0]))

            Parameters
            ----------
            x: (string, dictionary, dictionary)
            y: (string, dictionary, dictionary)

            Returns
            ----------
            res: float.
        """

        return self._phi(x).dot(self._phi(y))

    def _phi(self, x):
        """
            SpectrumKernel._phi
            Compute Phi(x). Unit version (slow).

            Parameters
            ----------
            x: string.

            Returns
            ----------
            res: np.array((self.lex_size**self.k)).
                Counts the number of substrings by index.
        """

        # Suppose the values will never go above uint16...
        # Otherwise, will have to use scipy.sparse
        phi_x = np.zeros(self.lex_size ** self.k, dtype=np.uint16)
        for l in range(len(x) - self.k + 1):
            phi_x[self.get_preindexed_value(x[l:l + self.k])] += 1
        # ~ print(phi_x)
        return phi_x

    def _phi_from_list(self, X, m):
        """
            SpectrumKernel._phi_from_list
            Compute Phi(x) for each x in X.

            Parameters
            ----------
            X: list(string) (length m).

            Returns
            ----------
            res: np.array((m, self.lex_size**self.k)).
                Each line counts the number of substrings by index in the
                corresponding string of the same index.
        """

        # l is the length of the strings
        # suppose they are all of the same length (otherwise, must do
        # some more complicated matrix operations)
        l = len(X[0])

        # Let the code in the dictionary be of length < 256...
        # Build the matrix Xtr_encoded such that Xtr_encoded[i, :] is
        # the ith training sample encoded with the dictionnary
        if self.lex_size >= 256:
            raise Exception("Number too big for uint8!")
        X_encoded = np.zeros((m, l), dtype=np.uint8)
        for i in range(m):
            X_encoded[i] = [self.lexicon[X[i][j]] for j in range(l)]

        # ~ print(Xtr_encoded)

        # Build a circulant matrix for computing the identifier of each
        # substring of length k
        # The identifier is sum(lex[x[0]] * lex_size**0 + lex[x[1]] * lex_size**1 ...)
        # For instance: if self.lex_size = 4 and self.k = 3:
        # [[ 1  4 16  0  0  0 ... ]
        #  [ 0  1  4 16  0  0 ... ]
        #  [ 0  0  1  4 16  0 ... ]]
        # of size (l-self.k+1, l)
        if self.lex_size ** self.k >= 65535:
            raise Exception("Number too big for uint16!")
        T = np.zeros((l - self.k + 1, l), dtype=np.uint16)
        # ~ print("T:", T.shape)
        for i in range(self.k):
            diag = np.diagonal(T, i)
            diag.setflags(write=True)
            diag.fill(self.lex_size ** i)

        # ~ print(T)

        # Create a matrix with (i, j) being the identifier of the substring
        # under consideration (simple product!)
        # For instance X_indexed[i, j] will be the identifier of substring j
        # in sample i.
        X_indexed = X_encoded.dot(T.T)

        # ~ print(X_indexed)

        # Now fill a sparse matrix of size (n, self.lex_size**self.k)
        # containing the counts of each substring
        # Assume the counts are always less than l-k+1 < 65535
        if l - self.k + 1 >= 65535:
            raise Exception("Number too big for uint16!")
        # Put a uint32 since this matrix will be multiplied by itself at
        # some point
        Phi = sparse.dok_matrix((m, self.lex_size ** self.k), dtype=np.uint32)
        for i in range(X_indexed.shape[0]):
            for j in range(X_indexed.shape[1]):
                Phi[i, X_indexed[i, j]] += 1

        return Phi.tocsr()

    def compute_K_train(self, Xtr, n, verbose=True):
        """
            SpectrumKernelPreindexed.compute_K_train
            Compute K from data.
            Overrides generic method.

            Parameters
            ----------
            Xtr: list(string).
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
            print("Called SpectrumKernelPreindexed.compute_K_train")
            start = time.time()

        l = len(Xtr[0])

        self.Phi_tr = self._phi_from_list(Xtr, n)

        # ~ print(self.Phi_tr)

        # Finally take the scalar product
        # Assume the counts squared are always less than (l-k+1)**2 < 65535
        if (l - self.k + 1) ** 2 >= 65535:
            raise Exception("Number too big for uint16!")
        K = np.array(self.Phi_tr.dot(self.Phi_tr.T).todense(), dtype=np.float64)

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            SpectrumKernelPreindexed.compute_K_test
            Gets the matrix K_t = [K(t_i, x_j)] where t_i is the ith test sample
            and x_j is the jth training sample.
            NON CENTERED KERNEL VALUES version. The centered version is defined
            in derived class CenteredKernel.

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
            print("Called SpectrumKernelPreindexed.compute_K_test")
            start = time.time()
        # ~ K_t = np.zeros((m, n))

        l = len(Xte[0])

        self.Phi_te = self._phi_from_list(Xte, m)

        # Finally take the scalar product
        # Assume the counts squared are always less than (l-k+1)**2 < 65535
        if (l - self.k + 1) ** 2 >= 65535:
            raise Exception("Number too big for uint16!")
        K_t = np.array(self.Phi_te.dot(self.Phi_tr.T).todense(), dtype=np.float64)

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_t