# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:56:16 2018


Implements kernels and associated functions. A kernel instance should be an
object with methods:
    evaluate(self, x, y): get kernel value for points (x, y).
    compute_matrix_K(self, Xtr, n): get training kernel Gram matrix.
    get_test_K_evaluations(self, Xtr, n, Xte, m, verbose=True): get train
        vs test kernel Gram matrix.


@author: Quentin
"""


from abc import ABC, abstractmethod
import numpy as np


#%%


class Kernel(ABC):
    
    
    @abstractmethod
    def evaluate(self, x, y):
        pass
    
    
    """
        Kernel.get_test_K_evaluations
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
            print("Called get_test_K_evaluations (NON CENTERED VERSION)")
        
        K_t = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                K_t[i, j] = self.evaluate(Xte[i], Xtr[j])
        
        if verbose:
            print("end")
        
        return K_t
    
    
    """
        Kernel.compute_matrix_K
        Compute K from data.
        
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
    def compute_matrix_K(self, Xtr, n):
        K = np.zeros([n, n], dtype=float)
        for i in range(n):
            print(i)
            K[i, i] = self.evaluate(Xtr[i], Xtr[i])
            for j in range(i):
                K[i, j] = self.evaluate(Xtr[i], Xtr[j])
                K[j, i] = K[i, j]
        return K


#%%
    

"""
    CenteredKernel
    Class that implements computing centered kernel values.
"""
class CenteredKernel(Kernel):
    
    def __init__(self, kernel):
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
    def train(self, Xtr, n, K=None):
        self.Xtr = Xtr
        self.n = n
        if K is None:
            # Compute the non-centered Gram matrix
            self.K = self.kernel.compute_matrix_K(self.Xtr, self.n)
        else:
            self.K = K
        
        self.eta = np.sum(self.K) / np.power(self.n, 2)
        
        # Store centered kernel
        U = np.ones(self.K.shape) / self.n
        self.centered_K = (np.eye(self.n) - U).dot(self.K).dot(np.eye(self.n) - U)

    
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
        
        # Get the non-centered K test
        K_t_nc = self.kernel.get_test_K_evaluations(Xtr, n, Xte, m, verbose)
    
        # The new K_t is the non-centered matrix
        #   + 1/n * sum_l(K_{k, l}) where k is test index and l is train index
        #   + 1/n * sum_l(K_{l, j}) where l and j are train indices
        #   + 1/n^2 * sum_{l, l'}(K_{l, l'}) where l, l' are train indices
        #   (the latter quantity was stored in self.eta)
        K_t = K_t_nc + (-1/self.n) * ( \
                K_t_nc.dot(np.ones((n,n))) + \
                np.ones((m, n)).dot(self.K))
        K_t += self.eta
        
        if verbose:
            print("end")
        
        return K_t


#%%


"""
    Linear_kernel
"""
class Linear_kernel(Kernel):
    
    
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


"""
    Gaussian_kernel
"""
class Gaussian_kernel(Kernel):

    
    def __init__(self, gamma):
        self.gamma = gamma
    
    
    """
        Gaussian_kernel.evaluate
        Compute exp(- gamma * norm(x, y)^2).
        
        Parameters
        ----------
        x: np.array.
        y: np.array.
        
        Returns
        ----------
        res: float.
    """
    def evaluate(self, x, y):
        return np.exp(-self.gamma * np.sum(np.power(x - y, 2))) 


"""
    Spectrum_kernel
    STILL NEEDS SOME SPEED UP
"""
class Spectrum_kernel(Kernel):
    
    def __init__(self, k,EOW = '$'):
        self.k = k
        self.EOW = '$'
        
    """
        Spectrum_kernel.evaluate
        Compute Phi(x[0]).dot(Phi(y[0]))
        where Phi_u(x[0]) denotes the number of occurences of u in sequence x[0]
        u in {A,T,C,G}^k
        x[0] = sequence
        x[1] = trie
        
        Parameters
        ----------
        x: (string, dictionary)
        y: (string, dictionary)
        
        Returns
        ----------
        res: float.
    """
    def evaluate(self, x, y):
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
        yphi = np.zeros(count)
        # this last part probably takes too long, 
        # maybe precompute the number of occurences outside of the kernel
        # (similarly to the computation of the tries)?
        for (i,w) in zip(range(len(xwords.keys())),xwords.keys()):
            yphi[i] = sum(y[0][j:].startswith(w) for j in range(len(y[0])))
        
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
    
