# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:56:16 2018


Implements kernels and associated functions. A kernel instance should be an
object with methods:
    evaluate(self, x, y): get kernel value for points (x, y).
    compute_matrix_K(self, Xtr, n): get training kernel Gram matrix.
    get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True): get train
        vs test kernel Gram matrix.


Multithreading capabilities:
    If the types of the object enables it (e.g. gaussian or laplacian
    kernels), use numba to jit and parallelize the operations.
    Otherwise, use joblib to multithread.


@author: Quentin
"""


from abc import ABC, abstractmethod
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import multiprocessing
import joblib as jl
import numba

import time


#%%


class Kernel(ABC):
    
    
    def __init__(self, enable_joblib=False):
        self.enable_joblib = enable_joblib
    
    
    @abstractmethod
    def evaluate(self, x, y):
        pass
    
    
    
    def _fill_train_line(self, Xtr, i):
        res = np.zeros((i+1,))
        for j in range(i+1):
            res[j] = self.evaluate(Xtr[i], Xtr[j])
        return res
    
    
    """
        Kernel.compute_matrix_K
        Compute K from data.
        JOBLIB GENERIC VERSION
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        n: int.
            Length of Xtr.
        
        Returns
        ----------
        K: np.array.
    """
    def compute_matrix_K(self, Xtr, n, verbose=True):
        
        if verbose:
            print("Called compute_matrix_K (NON CENTERED VERSION)")
            start = time.time()
        
        
        K = np.zeros([n, n], dtype=np.float64)

        
        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", \
                          multiprocessing.cpu_count())
            
            # Better results when processing from the larger to the shorter
            # line
            results = jl.Parallel(n_jobs=multiprocessing.cpu_count()) (\
                        jl.delayed(self._fill_train_line)(Xtr, i) \
                        for i in range(n-1, -1, -1) \
                        )
            for i in range(n):
                K[:i+1, i] = results[n-1-i]
        else:
            for i in range(n):
                K[:i+1, i] = self._fill_train_line(Xtr, i)
        
        
        # Symmetrize
        K = K + K.T - np.diag(K.diagonal())
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end-start))

        return K
    
    
    
    def _fill_test_column(self, Xte, Xtr, j, m):
        res = np.zeros((m,))
        for k in range(m):
            res[k] = self.evaluate(Xte[k], Xtr[j])
        return res
    
    
    """
        Kernel.get_test_K_evaluations
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
            
        Returns
        ----------
        K_t: np.array (shape=(m,n)).
    """
    def get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True):
        
        
        if verbose:
            print("Called get_test_K_evaluations (NON CENTERED VERSION)")
            start = time.time()
        K_t = np.zeros((m, n))
        
        
        if self.enable_joblib:
            if verbose:
                print("Called joblib loop on n_jobs=", \
                          multiprocessing.cpu_count())
            
            results = jl.Parallel(n_jobs=multiprocessing.cpu_count())(\
                        jl.delayed(self._fill_test_column)(Xte, Xtr, j, m) \
                        for j in range(n)\
                        )
            for j in range(n):
                K_t[:,j] = results[j]
        else:
            for j in range(n):
                K_t[:,j] = self._fill_test_column(Xte, Xtr, j, m)
        
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end-start))
        
        return K_t




#%%
    

"""
    CenteredKernel
    Class that implements computing centered kernel values.
"""
class CenteredKernel(Kernel):
    
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
            self.K = self.kernel.compute_matrix_K(self.Xtr, self.n, \
                                                  verbose=verbose)
        else:
            self.K = K
        
        self.eta = np.sum(self.K) / np.power(self.n, 2)
        
        # Store centered kernel
        U = np.ones(self.K.shape) / self.n
        self.centered_K = (np.eye(self.n) - U).dot(self.K)\
                            .dot(np.eye(self.n) - U)

    
    """
        CenteredKernelernel.get_centered_K
        Gets the centered Gram matrix.
        
        Returns
        ----------
        centered_K: np.array.
    """
    def get_centered_K(self):
        return self.centered_K
    
    
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
    def evaluate(self, x, y):
        tmp = 0.0
        for k in range(self.n):
            tmp += self.kernel.evaluate(x, self.Xtr[k])
            tmp += self.kernel.evaluate(y, self.Xtr[k])
        #print("tmp:", tmp)
        #print("tmp/n:", tmp / self.n)
        #print("centered")
        return self.kernel.evaluate(x, y) - tmp / self.n + self.eta
    
    """
        CenteredKernel.get_test_K_evaluations
        Compute centered values of the matrix K_t = [K(t_i, x_j)] where t_i is 
        the ith test sample and x_j is the jth training sample.
        CENTERED KERNEL VALUES version. 
        Overrides Kernel.get_test_K_evaluations.
        
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
            
        Returns
        ----------
        K_t: np.array (shape=(m,n)).
    """
    def get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True):
        
        if verbose:
            print("Called get_test_K_evaluations (CENTERED VERSION)")
            start = time.time()
        
        # Get the non-centered K test
        K_t_nc = self.kernel.get_test_K_evaluations(Xtr, n, Xte, m, verbose)
    
        # The new K_t is the non-centered matrix
        #   + 1/n * sum_l(K_{k, l}) where k is test index and l is train index
        #   + 1/n * sum_l(K_{l, j}) where l and j are train indices
        #   + 1/n^2 * sum_{l, l'}(K_{l, l'}) where l, l' are train indices
        #   (the latter quantity was stored in self.eta)
        K_t = K_t_nc + (-1/self.n) * ( \
                K_t_nc.dot(np.ones((self.n,self.n))) + \
                np.ones((m, self.n)).dot(self.K))
        K_t += self.eta
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end-start))
        
        return K_t


#%%


"""
    Linear_kernel
"""
class Linear_kernel(Kernel):
    
    
    def __init__(self, enable_joblib=False):
        super().__init__(enable_joblib)
  
    
    """
        Linear_kernel.evaluate
        Compute x.dot(y)
        
        Parameters
        ----------
        x: np.array.
        y: np.array.
        
        Returns
        ----------
        res: float.
    """
    def evaluate(self, x, y):
        return x.dot(y.transpose())


@numba.jit(nopython=True, nogil=True, cache=True)
def _jit_ev_gaussian(x, y, gamma):
    return np.exp(-gamma * np.sum(np.power(x - y, 2))) 


@numba.jit(nopython=True, parallel=True, nogil=True)
def _jit_Ktr_gaussian(Xtr, n, gamma):
    K = np.zeros((n, n), dtype=np.float64)

    for i in numba.prange(n):
        res = np.zeros((i+1,))
        for j in numba.prange(i+1):
            res[j] = _jit_ev_gaussian(Xtr[i], Xtr[j], gamma)
        K[:i+1, i] = res
        
    # Symmetrize
    return K + K.T - np.diag(np.diag(K))


@numba.jit(nopython=True, parallel=True, nogil=True)
def _jit_Kte_gaussian(Xtr, n, Xte, m, gamma):
    K_t = np.zeros((m, n), dtype=np.float64)
        
    for j in numba.prange(n):
        res = res = np.zeros((m,), dtype=np.float64)
        for k in numba.prange(m):
            res[k] = _jit_ev_gaussian(Xte[k], Xtr[j], gamma)
        K_t[:,j] = res
    
    return K_t


"""
    Gaussian_kernel
"""
class Gaussian_kernel(Kernel):

    
    def __init__(self, gamma, enable_joblib=False):
        super().__init__(enable_joblib)
        self.gamma = gamma
    
    
    """
        Gaussian_kernel.evaluate
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
    def evaluate(self, x, y):
        return _jit_ev_gaussian(x, y, self.gamma)
    
    
    """
        Gaussian_kernel.compute_matrix_K
        Compute K from data.
        JIT VERSION - OVERRIDES PARENT METHOD
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        n: int.
            Length of Xtr.
        
        Returns
        ----------
        K: np.array.
    """
    @numba.jit(cache=True)
    def compute_matrix_K(self, Xtr, n, verbose=True):
        if verbose:
            print("Gaussian_kernel.compute_matrix_K")
            res = _jit_Ktr_gaussian(Xtr, n, self.gamma)
            print("end")
        return res
    
    
    
    """
        Gaussian_kernel.get_test_K_evaluations
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
            
        Returns
        ----------
        K_t: np.array (shape=(m,n)).
    """
    @numba.jit(cache=True)
    def get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True):
        
        if verbose:
            print("Gaussian_kernel.get_test_K_evaluations")
            
        K_t = _jit_Kte_gaussian(Xtr, n, Xte, m, self.gamma)
        
        if verbose:
            print("end")
        
        return K_t


"""
    Spectrum_kernel
    STILL NEEDS SOME SPEED UP
"""
class Spectrum_kernel(Kernel):
    
    def __init__(self, k,EOW = '$', enable_joblib=False):
        super().__init__(enable_joblib)
        self.k = k
        self.EOW = '$'
        
    """
        Spectrum_kernel.evaluate
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
    def evaluate(self, x, y):
        #print("in evaluate")
        xwords = {}
        count = 0
        for l in range(len(x[0])-self.k+1):
            if self.find_kmer(y[1],x[0][l:l+self.k]):
                if x[0][l:l+self.k] in xwords.keys():
                    xwords[x[0][l:l+self.k]] += 1
                else:
                    xwords[x[0][l:l+self.k]] = 1
                    count += 1
        xphi = np.fromiter(xwords.values(), dtype=int)
        yphi = [y[2][key] for key in xwords.keys()]
        # this last part probably takes too long, 
        # maybe precompute the number of occurences outside of the kernel
        # (similarly to the computation of the tries)?
        #for (i,w) in zip(range(len(xwords.keys())),xwords.keys()):
        #    yphi[i] = sum(y[0][j:].startswith(w) for j in range(len(y[0])))
        
        return xphi.dot(yphi)
                     

    """
        Spectrum_kernel.find_kmer
        Finds whether a word kmer is present in a given retrieval tree trie
    """
    def find_kmer(self,trie,kmer):
        tmp = trie
        for l in kmer:
            if l in tmp:
                tmp = tmp[l]
            else:
                return False
        return self.EOW in tmp
    

"""
    Spectrum_kernel_DNA_preindexed
"""
class Spectrum_kernel_preindexed(Kernel):
    
    
    """
        Spectrum_kernel_preindexed.__init__
        
        Parameters
        ----------
        k: int. Length of the substrings to be looked for.
        lexicon: dict. Map key to integer.
    """
    def __init__(self, k, lexicon, enable_joblib=False):
        super().__init__(enable_joblib)
        self.k = k
        self.lexicon = lexicon
        self.lex_size = len(lexicon)
        #~ print(lexicon)
        #~ print(self.lex_size)
    
    
    """ 
        Spectrum_kernel_preindexed.get_index
    
        Preindex lexicon, for instance a -> 1, b -> 2... z -> 26
        Here we work with DNA sequences, thus the lexicon is shallow
        (i.e. size lex_size=4). We take as the computed index simply 
        the sum in base lex_size (i.e. for a sequence abcd of length
        k=4 we would have 0 + 1 * 4 + 2 * 4**2 + 3 * 4**3, i.e. an
        indexing on 256 values. We can go reasonably far (indices grow
        with 4^{k-1}, we are limited, I believe, to int32).
        We could concatenate arrays for combining values l < k,
        until some point (the indices grow with (4**k - 1 / 3)).
    """
    def get_preindexed_value(self, lookup_str):
        res = 0
        for i in range(self.k):
            res += self.lexicon[lookup_str[i]] * self.lex_size**i
        #print(lookup_str, res)
        return res
        
    """
        Spectrum_kernel.evaluate
        Compute Phi(x[0]).dot(Phi(y[0]))
        
        Parameters
        ----------
        x: (string, dictionary, dictionary)
        y: (string, dictionary, dictionary)
        
        Returns
        ----------
        res: float.
    """
    def evaluate(self, x, y):
        return self._phi(x).dot(self._phi(y))
    
    
    """
        Spectrum_kernel._phi
        Compute Phi(x). Unit version (slow).
        
        Parameters
        ----------
        x: string.
        
        Returns
        ----------
        res: np.array((self.lex_size**self.k)).
            Counts the number of substrings by index.
    """
    def _phi(self, x):
        
        # Suppose the values will never go above uint16...
        # Otherwise, will have to use scipy.sparse
        phi_x = np.zeros(self.lex_size**self.k, dtype=np.uint16)
        for l in range(len(x)-self.k+1):
            phi_x[self.get_preindexed_value(x[l:l+self.k])] += 1
        #~ print(phi_x)
        return phi_x
        
    
    """
        Spectrum_kernel._phi_from_list
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
    def _phi_from_list(self, X, m):
        
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
        
        #~ print(Xtr_encoded)
        
        # Build a circulant matrix for computing the identifier of each
        # substring of length k
        # The identifier is sum(lex[x[0]] * lex_size**0 + lex[x[1]] * lex_size**1 ...)
        # For instance: if self.lex_size = 4 and self.k = 3:
        # [[ 1  4 16  0  0  0 ... ]
        #  [ 0  1  4 16  0  0 ... ]
        #  [ 0  0  1  4 16  0 ... ]]
        # of size (l-self.k+1, l)
        if self.lex_size**self.k >= 65535:
            raise Exception("Number too big for uint16!")
        T = np.zeros((l-self.k+1, l), dtype=np.uint16)
        #~ print("T:", T.shape)
        for i in range(self.k):
            diag = np.diagonal(T, i)
            diag.setflags(write=True)
            diag.fill(self.lex_size**i)
        
        #~ print(T)
        
        # Create a matrix with (i, j) being the identifier of the substring
        # under consideration (simple product!)
        # For instance X_indexed[i, j] will be the identifier of substring j
        # in sample i.
        X_indexed = X_encoded.dot(T.T)
        
        #~ print(X_indexed)
        
        # Now fill a sparse matrix of size (n, self.lex_size**self.k)
        # containing the counts of each substring
        # Assume the counts are always less than l-k+1 < 65535
        if l-self.k+1 >= 65535:
            raise Exception("Number too big for uint16!")
        # Put a uint32 since this matrix will be multiplied by itself at
        # some point
        Phi = sparse.dok_matrix((m, self.lex_size**self.k), dtype=np.uint32) 
        for i in range(X_indexed.shape[0]):
            for j in range(X_indexed.shape[1]):
                Phi[i, X_indexed[i,j]] += 1
        
        return Phi.tocsr()
    
    """
        Spectrum_kernel_preindexed.compute_matrix_K
        Compute K from data.
        Overrides generic method.
        
        Parameters
        ----------
        Xtr: list(string). 
            Training data.
        n: int.
            Length of Xtr.
        
        Returns
        ----------
        K: np.array.
    """
    def compute_matrix_K(self, Xtr, n, verbose=True):
        
        if verbose:
            print("Called Spectrum_kernel_preindexed.compute_matrix_K")
            start = time.time()
        
        l = len(Xtr[0])
        
        self.Phi_tr = self._phi_from_list(Xtr, n)
        
        #~ print(self.Phi_tr)
        
        # Finally take the scalar product
        # Assume the counts squared are always less than (l-k+1)**2 < 65535
        if (l-self.k+1)**2 >= 65535:
            raise Exception("Number too big for uint16!")
        K = np.array(self.Phi_tr.dot(self.Phi_tr.T).todense(), dtype=np.float64)

        return K

    
    """
        Spectrum_kernel_preindexed.get_test_K_evaluations
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
            
        Returns
        ----------
        K_t: np.array (shape=(m,n)).
    """
    def get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True):
        
        if verbose:
            print("Called Spectrum_kernel_preindexed.get_test_K_evaluations")
            start = time.time()
        #~ K_t = np.zeros((m, n))
        
        l = len(Xte[0])
        
        self.Phi_te = self._phi_from_list(Xte, m)
        
        # Finally take the scalar product
        # Assume the counts squared are always less than (l-k+1)**2 < 65535
        if (l-self.k+1)**2 >= 65535:
            raise Exception("Number too big for uint16!")
        K_t = np.array(self.Phi_te.dot(self.Phi_tr.T).todense(), dtype=np.float64)
        
        if verbose:
            end = time.time()
            print("end. Time elapsed:", "{0:.2f}".format(end-start))
        
        return K_t
