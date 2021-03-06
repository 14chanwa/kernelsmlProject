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
import time
import numba
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
        SpectrumKernelPreindexed
    """

    def __init__(self, k, lexicon, remove_dimensions=False, normalize=False, enable_joblib=False):
        """
            SpectrumKernelPreindexed.__init__

            Parameters
            ----------
            k: int. 
                Length of the substrings to be looked for.
            lexicon: dict. 
                Map key to integer.
            enable_joblib: bool.
        """

        super().__init__(enable_joblib)
        self.k = k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        self.normalize = normalize
        self.remove_dimensions = remove_dimensions

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
            SpectrumKernelPreindexed.evaluate
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
            SpectrumKernelPreindexed._phi
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
            SpectrumKernelPreindexed._phi_from_list
            Compute Phi(x) for each x in X.

            Parameters
            ----------
            X: list(string) (length m).
            m: int.
                Length of X.

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
        if self.lex_size ** self.k >= 4294967295:
            raise Exception("Number too big for uint32!")
        T = np.zeros((l - self.k + 1, l), dtype=np.uint32)
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
        #~ # Assume the counts are always less than l-k+1 < 65535
        #~ if l - self.k + 1 >= 65535:
            #~ raise Exception("Number too big for uint16!")
        #~ # Put a uint32 since this matrix will be multiplied by itself at
        # some point
        Phi = sparse.lil_matrix((m, self.lex_size ** self.k), dtype=np.float64)
        # TODO: is there a way to speed this double loop up?
        # A solution would be to use np.bincount... but this imply using
        # nonsparse vectors. Yet this seems tractable and fast (~1-10s)
        # Trying to use vectorized operations, like Phi[i, X_indexed[i, :]] += 1
        # do not work since then each index would be counted once, even if it
        # appears multiple times.
        # for i in range(X_indexed.shape[0]):
            # for j in range(X_indexed.shape[1]):
                # Phi[i, X_indexed[i, j]] += 1
        for i in range(X_indexed.shape[0]):
            tmp = np.bincount(X_indexed[i, :])
            Phi[i, :tmp.size] += tmp

        return Phi

    def compute_K_train(self, Xtr, n, verbose=True):
        """
            SpectrumKernelPreindexed.compute_K_train
            Compute K from data.

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
        
        if self.remove_dimensions:
        
            self.kept_columns = self.Phi_tr.getnnz(0)>0
            
            # Only keep nonzero columns
            self.Phi_tr = self.Phi_tr[:,self.kept_columns]
            if sparse.isspmatrix(self.Phi_tr):
                self.Phi_tr = self.Phi_tr.todense()
            
            if self.normalize:
                
                # Compute mu and sigma
                self.mus = np.mean(self.Phi_tr, 0)
                self.sigmas = np.std(self.Phi_tr, 0)
                # Center and normalize
                self.Phi_tr -= self.mus
                self.Phi_tr /= self.sigmas
                
                #~ self.Phi_tr = sparse.csr_matrix(self.Phi_tr)
            

            # ~ print(self.Phi_tr)

            # Finally take the scalar product
            # Assume the counts squared are always less than (l-k+1)**2 < 65535
            if (l - self.k + 1) ** 2 >= 65535:
                raise Exception("Number too big for uint16!")
            K = np.array(self.Phi_tr.dot(self.Phi_tr.T), dtype=np.float64)
        
        else:
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
        
        if self.remove_dimensions:
        
            # Only keep nonzero columns at time of train
            self.Phi_te = self.Phi_te[:,self.kept_columns]
            if sparse.isspmatrix(self.Phi_te):
                self.Phi_te = self.Phi_te.todense()
            
            if self.normalize:
                
                # Center and normalize
                self.Phi_te -= self.mus
                self.Phi_te /= self.sigmas
                
                #~ self.Phi_te = sparse.csr_matrix(self.Phi_te)

            # Finally take the scalar product
            # Assume the counts squared are always less than (l-k+1)**2 < 65535
            if (l - self.k + 1) ** 2 >= 65535:
                raise Exception("Number too big for uint16!")
            K_t = np.array(self.Phi_te.dot(self.Phi_tr.T), dtype=np.float64)
        
        else:
            K_t = np.array(self.Phi_te.dot(self.Phi_tr.T).todense(), dtype=np.float64)
            
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_t


########################################################################
### MultipleSpectrumKernel                                                         
########################################################################


class MultipleSpectrumKernel(Kernel):
    """
        MultipleSpectrumKernel
        Let k_1, k_2... be possible lengths of consecutive substrings.
        Let Phi_{k_1}, Phi_{k_2}... be the transforms by the Spectrum
        kernel ; then define Phi associated to this kernel as the
        sum of all these values. The associated K is indeed a positive
        definite function, so a valid kernel.
    """

    def __init__(self, list_k, lexicon, remove_dimensions=False, normalize=False, enable_joblib=False):
        """
            MultipleSpectrumKernel.__init__

            Parameters
            ----------
            list_k: list(int). 
                List of k such that the kernel is the sum of the Phi_k.
            lexicon: dict. 
                Map key to integer.
        """

        super().__init__(enable_joblib)
        self.list_k = list_k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        self.normalize = normalize
        self.remove_dimensions = remove_dimensions
        
        # Create a list of kernels to be evaluated
        self.list_kernels = []
        for i in range(len(self.list_k)):
            self.list_kernels.append(SpectrumKernelPreindexed(self.list_k[i], self.lexicon, remove_dimensions=self.remove_dimensions, normalize=self.normalize))
    
    def evaluate(self, x, y):
        """
            MultipleSpectrumKernel.evaluate
            Compute Phi(x[0]).dot(Phi(y[0]))

            Parameters
            ----------
            x: (string, dictionary, dictionary)
            y: (string, dictionary, dictionary)

            Returns
            ----------
            res: float.
        """
        
        res = 0
        for kernel in self.list_kernels:
            res += kernel.evaluate(x, y)
        
        return res
    
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            MultipleSpectrumKernel.compute_K_train
            Compute K from data.

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
            print("Called MultipleSpectrumKernel.compute_K_train")
            start = time.time()

        K = np.zeros((n, n))
        for kernel in self.list_kernels:
            K += kernel.compute_K_train(Xtr, n, verbose)
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            MultipleSpectrumKernel.compute_K_test
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
            K_t: np.array (shape=(m,n)).
        """

        if verbose:
            print("Called MultipleSpectrumKernel.compute_K_test")
            start = time.time()
        
        K_t = np.zeros((m, n))
        for kernel in self.list_kernels:
            K_t += kernel.compute_K_test(Xtr, n, Xte, m, verbose)

        return K_t



########################################################################
### MultipleSpectrumKernel_Gaussian                                                       
########################################################################

@numba.jit(nopython=True, parallel=True, nogil=True)
def _add_all_squares(Phi_i, Phi_j):
    res = np.zeros((Phi_i.shape[0], Phi_j.shape[0]))
    # Add (Phi[i] - Phi[j])^2 to the term (i, j)
    for i in numba.prange(Phi_i.shape[0]):
        for j in numba.prange(Phi_j.shape[0]):
            u = Phi_i[i,:] - Phi_j[j,:]
            res[i, j] += np.dot(u, u)
    return res


@numba.jit(nopython=True, parallel=True, nogil=True)
def _add_all_squares_sym(Phi):
    res = np.zeros((Phi.shape[0], Phi.shape[0]))
    # Add (Phi[i] - Phi[j])^2 to the term (i, j)
    for i in numba.prange(Phi.shape[0]):
        for j in numba.prange(i):
            u = Phi[i,:] - Phi[j,:]
            res[i, j] += np.dot(u, u)
            res[j, i] += res[i, j]
    return res


class MultipleSpectrumGaussianKernel(Kernel):
    """
        MultipleSpectrumGaussianKernel
        Let k_1, k_2... be possible lengths of consecutive substrings.
        Let Phi_{k_1}, Phi_{k_2}... be the transforms by the Spectrum
        kernel ; then define Phi associated to this kernel as the
        sum of all these values. 
        Stash Gaussian kernel on top of the concatenated Phis
    """

    def __init__(self, list_k, lexicon, gamma, normalize=False, enable_joblib=False):
        """
            MultipleSpectrumGaussianKernel.__init__

            Parameters
            ----------
            list_k: list(int). 
                List of k such that the kernel is the sum of the Phi_k.
            lexicon: dict. 
                Map key to integer.
            gamma: float.
                The parameter of the gaussian kernel on top of the Phis.
        """

        super().__init__(enable_joblib)
        self.list_k = list_k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        
        self.normalize = normalize
        
        # Gaussian parameter
        self.gamma = gamma
        
        # Kept columns
        self.kept_columns = []
        # For normalization...
        self.mus = []
        self.sigmas = []
        self.total_nb_dims = 0
        
        # Create a list of kernels to be evaluated
        self.list_kernels = []
        for i in range(len(self.list_k)):
            self.list_kernels.append(SpectrumKernelPreindexed(self.list_k[i], self.lexicon))
    
    def evaluate(self, x, y):
        """
            MultipleSpectrumGaussianKernel.evaluate
            Compute K_gaussian (Phi(x), (Phi(y)))

            Parameters
            ----------
            x: (string, dictionary, dictionary)
            y: (string, dictionary, dictionary)

            Returns
            ----------
            res: float.
        """
        
        res = 0
        for kernel in self.list_kernels:
            u = kernel._phi(x) - kernel._phi(y)
            res += u.dot(u)
        res = np.exp(-self.gamma * res)
        
        return res
    
    #~ def _add_all_squares_j(self, Phi_i, Phi, ind_max):
        #~ res = np.zeros((1, Phi.shape[0]))
        #~ # Add (Phi[i] - Phi[j])^2 to the term (i, j)
        #~ for j in range(ind_max-1):
            #~ u = Phi_i.astype(dtype=np.int32, casting='unsafe', copy=False) - Phi[j,:].astype(dtype=np.int32, casting='unsafe', copy=False)
            #~ res[0, j] += u.dot(u.transpose())[0, 0]
        #~ return res
    
    @numba.jit(cache=True)
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            MultipleSpectrumGaussianKernel.compute_K_train
            Compute K from data.

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
            print("Called MultipleSpectrumGaussianKernel.compute_K_train")
            start = time.time()

        res = np.zeros((n, n))
        for k in range(len(self.list_kernels)):
            kernel = self.list_kernels[k]
            # Compute Phi
            Phi = kernel._phi_from_list(Xtr, n)
            # Convert to dense by removing zero columns
            self.kept_columns.append(Phi.getnnz(0)>0)
            Phi = Phi[:,self.kept_columns[k]]
            Phi = Phi.astype(np.float64).todense()
            self.total_nb_dims += Phi.shape[1]
            
            # Center
            self.mus.append(np.mean(Phi, 0))
            Phi -= self.mus[k]
            
            # Normalize
            if self.normalize:
                self.sigmas.append(np.std(Phi, 0))
                Phi /= self.sigmas[k]
            # Compute sum square differences
            res += _add_all_squares_sym(Phi)
        
        res /= self.total_nb_dims
        
        # Gaussianize
        K = np.exp(-self.gamma * res)
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K

    @numba.jit(cache=True)
    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            MultipleSpectrumGaussianKernel.compute_K_test
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
            K_t: np.array (shape=(m,n)).
        """

        if verbose:
            print("Called MultipleSpectrumGaussianKernel.compute_K_test")
            start = time.time()
        
        res = np.zeros((m, n))
        for k in range(len(self.list_kernels)):
            kernel = self.list_kernels[k]
            # Compute Phi
            Phi_i = kernel._phi_from_list(Xte, m).tocsr()
            Phi_j = kernel._phi_from_list(Xtr, n).tocsr()
            # Convert to dense by removing zero columns
            Phi_i = Phi_i[:, self.kept_columns[k]].astype(np.float64).todense()
            Phi_j = Phi_j[:, self.kept_columns[k]].astype(np.float64).todense()
            
            Phi_i -= self.mus[k]
            Phi_j -= self.mus[k]
            if self.normalize:
                Phi_i /= self.sigmas[k]
                Phi_j /= self.sigmas[k]
            # Compute sum square differences
            res += _add_all_squares(Phi_i, Phi_j)
        
        res /= self.total_nb_dims
        
        # Gaussianize
        K_t = np.exp(-self.gamma * res)

        return K_t



########################################################################
### MultipleSpectrumPolyKernel                                                         
########################################################################


class MultipleSpectrumPolyKernel(Kernel):
    """
        MultipleSpectrumPolyKernel
        Let k_1, k_2... be possible lengths of consecutive substrings.
        Let Phi_{k_1}, Phi_{k_2}... be the transforms by the Spectrum
        kernel ; then define Phi associated to this kernel as the
        sum of all these values. The associated K is indeed a positive
        definite function, so a valid kernel.
        Then, apply a polynomial kernel, i.e. K(x, y) <- (K(x, y) + c)**d
    """

    def __init__(self, list_k, lexicon, d=2, c=0, remove_dimensions=False, normalize=False):
        """
            MultipleSpectrumPolyKernel.__init__

            Parameters
            ----------
            list_k: list(int). 
                List of k such that the kernel is the sum of the Phi_k.
            lexicon: dict. 
                Map key to integer.
        """

        self.list_k = list_k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        self.normalize = normalize
        self.remove_dimensions = remove_dimensions
        self.d = d
        self.c = c
        
        # Create a list of kernels to be evaluated
        self.list_kernels = []
        for i in range(len(self.list_k)):
            self.list_kernels.append(SpectrumKernelPreindexed(self.list_k[i], self.lexicon, remove_dimensions=self.remove_dimensions, normalize=self.normalize))
    
    def evaluate(self, x, y):
        """
            MultipleSpectrumPolyKernel.evaluate
            Compute (sum{Phi(x[0]).dot(Phi(y[0]))} + c)**d

            Parameters
            ----------
            x: (string, dictionary, dictionary)
            y: (string, dictionary, dictionary)

            Returns
            ----------
            res: float.
        """
        
        res = 0
        for kernel in self.list_kernels:
            res += kernel.evaluate(x, y)
        
        return (res + self.c)**self.d
    
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            MultipleSpectrumKernel.compute_K_train
            Compute K from data.

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
            print("Called MultipleSpectrumPolyKernel.compute_K_train")
            start = time.time()

        K = np.zeros((n, n))
        for kernel in self.list_kernels:
            K += kernel.compute_K_train(Xtr, n, verbose)
        
        K += self.c
        K = np.power(K, self.d)
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            MultipleSpectrumPolyKernel.compute_K_test
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
            K_t: np.array (shape=(m,n)).
        """

        if verbose:
            print("Called MultipleSpectrumPolyKernel.compute_K_test")
            start = time.time()
        
        K_t = np.zeros((m, n))
        for kernel in self.list_kernels:
            K_t += kernel.compute_K_test(Xtr, n, Xte, m, verbose)
        
        K_t += self.c
        K_t = np.power(K_t, self.d)

        return K_t



########################################################################
### SubstringKernel                                                         
########################################################################


class SubstringKernel(Kernel):
    """
        SubstringKernel
    """

    def __init__(self, k, lambd, lexicon, enable_joblib=False):
        """
            SubstringKernel.__init__

            Parameters
            ----------
            k: int. Length of the substrings to be looked for.
            lambd: float. Parameter of substring kernel
            lexicon: dict. 
                Map key to integer.
        """

        super().__init__(enable_joblib)
        self.k = k
        self.lambd = lambd
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        
    
    def evaluate(self, x, y):
        """
            SubstringKernel.evaluate
            K(x,y) = sum_u phi_u(x)*phi_u(y)

            Parameters
            ----------
            x: string.
            y: string.

            Returns
            ----------
            res: float.
        """
        m = len(x)
        n = len(y)
        
        self.B = np.tile(-1,[m,n,self.k-1])
        self.b = np.tile(-1,[m,n,self.k-1])

        
        return self._K(x,y,self.k)
    
    
    def _K(self, x, y, i):
        m = len(x)
        n = len(y)

        if i == 0:
            return 1
        
        if min(m,n) < i:
            return 0
        
        s = x[-1]

        v = self.lambd*self._K(x[:m-1],y,i)
        for j in range(n):
            if y[j] == s:
                v += (self._B(x[:m-1],y[:j],i-1)*self.lambd**2)
        return v
    
    
    def _B(self, x, y, i):
        m = len(x)
        n = len(y)
        if i == 0:
            return 1
        if min(m,n)< i:
            return 0
        
        if self.B[m-1,n-1,i-1] != -1:
            return self.B[m-1,n-1,i-1]
        
        # Recursion
        
        v = self.lambd*(self._B(x[:m-1],y,i)+self._B(x,y[:n-1],i)) - self.lambd**2*self._B(x[:m-1],y[:n-1],i)
        if x[-1]==y[-1]:
            v += self.lambd**2*self._B(x[:m-1],y[:n-1],i-1)
        #v += self._b(x,y,i)
        
        self.B[m-1,n-1,i-1] = v
        return v
    
    def encode_X(self,X,m):
        # l is the length of the strings
        # suppose they are all of the same length (otherwise, must do
        # some more complicated matrix operations)
        l = len(X[0])

        # Let the code in the dictionary be of length < 256...
        # Build the matrix Xtr_encoded such that Xtr_encoded[i, :] is
        # the ith training sample encoded with the dictionnary
        if self.lex_size >= 256:
            raise Exception("Number too big for uint8!")
        X_encoded = np.zeros((m,l), dtype=np.uint8)
        for i in range(m):
            X_encoded[i] = [self.lexicon[X[i][j]] for j in range(l)]
        return X_encoded
    
    
    def compute_K_train(self, Xtr, n, verbose=True):
        """
            SubstringKernel.compute_K_train
            Compute K from data.

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
            print("Called SubstringKernel.compute_K_train")
            start = time.time()


        Xtr_enc = self.encode_X(Xtr, n)

        K = np.zeros([n, n], dtype=np.float64)
         
        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", multiprocessing.cpu_count())

            # Better results when processing from the larger to the shorter
            # line
            results = jl.Parallel(n_jobs=multiprocessing.cpu_count())(
                    jl.delayed(self._fill_train_line)(Xtr_enc, i)
                    for i in range(n - 1, -1, -1)
                )
            for i in range(n):
                K[:i + 1, i] = results[n - 1 - i]
        else:
            for i in range(n):
                K[:i + 1, i] = self._fill_train_line(Xtr_enc, i)

        # Symmetrize
        K = K + K.T - np.diag(K.diagonal())

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K
    

    def compute_K_test(self, Xtr, n, Xte, m, verbose=True):
        """
            SubstringKernel.compute_K_test
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
            K_t: np.array (shape=(m,n)).
        """

        if verbose:
            print("Called SpectrumKernelPreindexed.compute_K_test")
            start = time.time()
            
        Xtr_enc = self.encode_X(Xtr, n)
        Xte_enc = self.encode_X(Xte, m)
        
        K_t = np.zeros((m, n))

        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", multiprocessing.cpu_count())

            results = jl.Parallel(n_jobs=multiprocessing.cpu_count())(
                    jl.delayed(self._fill_test_column)(Xte_enc, Xtr_enc, j, m)
                    for j in range(n)
                )
            for j in range(n):
                K_t[:, j] = results[j]
        else:
            for j in range(n):
                K_t[:, j] = self._fill_test_column(Xte_enc, Xtr_enc, j, m)

        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end - start))

        return K_t
    



########################################################################
### WDKernel (Weighted degree Kernel
###           from:     Large Scale Genomic Sequence SVM Classifiers
###                     (Sonnenburg et al., 2005)                                         
########################################################################

class WDKernel(Kernel):
    """
        WDKernel
    """

    def __init__(self, k, lexicon, enable_joblib=False):
        """
            WDKernel.__init__

            Parameters
            ----------
            k: int. Length of the substrings to be looked for.
            lexicon: dict. 
                Map key to integer.
        """

        super().__init__(enable_joblib)
        self.k = k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        
    
    def evaluate(self, x, y):
        """
            WDKernel.evaluate
            K(x,y) = 

            Parameters
            ----------
            x: string.
            y: string.

            Returns
            ----------
            res: float.
        """
        
        m = len(x)
        
        res = 0
        
        i = 0
        while (i < m):
            while (i < m) and (x[i]!=y[i]) :
                i += 1
            B = 0
            while  (i < m) and (x[i]==y[i]):
                B += 1
                i += 1
            if B>self.k:
                res += (3*B-self.k+1)/3
            else:
                res += B*(-B**2 + 3*self.k*B + 3*self.k + 1)/(3*self.k*(self.k+1))
            
        
        return res
