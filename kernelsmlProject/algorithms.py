# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:03:51 2018


Implements machine learning algorithms. Instances should have methods:
    train(self, Xtr, Ytr, n): trains the model according to training data.
    predict(self, Xte, m): predicts regression/classification values for test
        data.


@author: Quentin
"""


from abc import ABC, abstractmethod
import numpy as np
from cvxopt import matrix, solvers
from kernelsmlProject.kernels import Linear_kernel, CenteredKernel


#%%


"""
    AlgorithmInstance
    Base class for an instance of regression problem solver.
"""
class AlgorithmInstance(ABC):
    
    
    @abstractmethod
    def train(self, Xtr, Ytr, n):
        pass
    
    
    """
        AlgorithmInstance.predict
        Generic prediction function: given test data, return prediction Y.
        
        Parameters
        ----------
        Xte: list(object). 
            Test data.
        m: int. 
            Length of Xte.
        
        Returns
        ----------
        Y: np.array (shape=(m,))
        
    """
    def predict(self, Xte, m, K_t = None):
        if self.verbose:
            print("Predict...")
        if K_t is None:
            self.K_t = self.kernel.get_test_K_evaluations(self.Xtr, self.n, Xte, m,\
                                                 self.verbose)
        else:
            self.K_t = K_t
        Yte = self.K_t.dot(self.alpha.reshape((self.alpha.size,1))).reshape(-1)
        
        if self.verbose:
            print("end")
        
        return Yte


    """
        AlgorithmInstance.init_train
        Builds self.K or/and its centered version.
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        Ytr: np.array (shape=(n,)). 
            Training targets.
        n: int. 
            Length of Xtr.
        K: np.array (shape=(n,n)). 
            Kernel Gram matrix or None.
    """
    def init_train(self, Xtr, Ytr, n, K):
        
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.n = n
        
        if K is None:
            if self.verbose:
                print("Build K...")
            self.K = self.kernel.compute_matrix_K(self.Xtr, self.n, \
                                                  self.verbose)
            if self.verbose:
                print("end")
        else:
            self.K = K
        
        
        if self.center:
            if self.verbose:
                print("Center K...")
            
            # Major issue solved: do NOT center an already centered kernel
            try:
                self.centeredKernel
            except AttributeError:
                self.centeredKernel = CenteredKernel(self.kernel)
            
            # Train the centered kernel
            self.centeredKernel.train(self.Xtr, self.n, self.K, self.verbose)
            self.kernel = self.centeredKernel
            # replace K by centered kernel
            # it is not important to replace K since centering a centered 
            # kernel leaves it unchanged.. in principle
            self.K = self.centeredKernel.get_centered_K() 
            if self.verbose:
                print("end")
    
    
    """
        AlgorithmInstance.get_training_results
        Get training results (for overfitting evaluation).
        
        Returns
        ----------
        f: np.array (shape=(n,)). Training results.
    """
    def get_training_results(self):
        f = np.sign(self.K.dot(self.alpha.reshape((self.alpha.size, 1))))
        return f.reshape(-1)


#%%


"""
    RidgeRegression
    Implements ridge regression.
"""
class RidgeRegression(AlgorithmInstance):
    
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    
    """
        RidgeRegression.train
        Fits internal vector alpha given training data.
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        Ytr: np.array (shape=(n,)). 
            Training targets.
        n: int. 
            Length of Xtr.
        lambd=0.1: float. 
            Optional. Regularization parameter.
        K=None: np.array (shape=(n,n)). 
            Optional. Gram matrix if available.
        W=None: np.array (shape=(n,n), diagonal). 
            Optional. Weights.        
    """
    def train(self, Xtr, Ytr, n, lambd=0.1, K=None, W=None):
        
        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(Xtr, Ytr, n, K)
        
        
        if W is None:
            W = np.eye(self.n,dtype=float)
            W_sqrt = W
        else:
            W_sqrt = np.sqrt(W)
        
        tmp = np.linalg.inv(W_sqrt.dot(self.K).dot(W_sqrt) + \
                            lambd * self.n * np.eye(self.n))
        self.alpha = W_sqrt.dot(tmp).dot(W_sqrt).dot(Ytr)
    

#%%


"""
    LogisticRegression
    Implements logistic regression.
"""
class LogisticRegression(AlgorithmInstance):
    
    # Maybe better: center=True as default?
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    
    """
        LogisticRegression.train
        Fits internal vector alpha given training data.
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        Ytr: np.array (shape=(n,)). 
            Training targets.
        n: int. 
            Length of Xtr.
        lambd=1: float. 
            Optional. Regularization parameter.
        K=None: np.array (shape=(n,n)). 
            Optional. Gram matrix if available.       
    """
    def train(self, Xtr, Ytr, n, lambd = 1, K = None):


        self.lambd = lambd

        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(Xtr, Ytr, n, K)
        
        
        if self.verbose:
            print("Start IRLS")
        self.alpha = self.IRLS()


    """
        LogisticRegression.IRLS
        Solves the logistic optimization problem using IRLS.
        
        Parameters
        ----------
        precision = 1e-20: float.
        max_iter = 1000: int.
        
        Returns
        ----------
        alpha: np.array (shape=(n,)).
            Solution vector.
    """
    def IRLS(self, precision = 1e-20, max_iter = 1000):

        alpha = np.zeros(self.n,dtype = float)
        W = np.eye(self.n,dtype = float)
        P = np.eye(self.n, dtype = float)
        z = np.zeros(self.n,dtype=float)
        
        for i in range(self.n):
            P[i,i] = -self.sigmoid(-self.Ytr[i]*self.K[i,].dot(alpha))
            W[i,i] = self.sigmoid(self.Ytr[i]*self.K[i,].dot(alpha))*(-P[i,i])
            z[i] = self.K[i,].dot(alpha) - self.Ytr[i]*P[i,i]/W[i,i]
            
        #z = self.K.dot(alpha) - np.linalg.inv(W).dot(P).dot(self.Ytr)
        ridge_regression = RidgeRegression(center=False, verbose=False)
        err = 1e12
        old_err = 0
        iter = 0
        while (iter < max_iter) & (np.abs(err-old_err) > precision):
            print(iter)
            print(np.abs(err-old_err))
            old_err = err
            ridge_regression.train(self.Xtr, z, self.n, self.lambd, self.K, W)
            alpha = ridge_regression.alpha
            
            m = self.K.dot(alpha)
            for i in range(self.n):
                P[i,i] = -self.sigmoid(-self.Ytr[i]*m[i])
                W[i,i] = self.sigmoid(self.Ytr[i]*m[i])*self.sigmoid(-self.Ytr[i]*m[i])
                z[i] = m[i] - self.Ytr[i]*P[i,i]/W[i,i]
            err = self.distortion(alpha)
            if err - old_err > 1e-10:
                print("Distortion is going up!")
            iter += 1
        
        if self.verbose:
            print("IRLS converged in %d iterations" %iter)
        
        return alpha

        
    def log_loss(self,u):
        return np.log(1+np.exp(-u))
        
    def sigmoid(self,u):
        return 1/(1+np.exp(-u))
        
    def distortion(self, alpha):
        J = 0.5*self.lambd*alpha.dot(self.K.dot(alpha))
        for i in range(self.n):
            J += self.log_loss(self.Ytr[i]*self.K[i,].dot(alpha)) / self.n
        return J


#%%
    
"""
    SVM
    Implements Support Vector Machine.
"""
class SVM(AlgorithmInstance):
    
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    """
        SVM.train
        Fits internal vector alpha given training data.
        
        Parameters
        ----------
        Xtr: list(object). 
            Training data.
        Ytr: np.array (shape=(n,)). 
            Training targets.
        n: int. 
            Length of Xtr.
        lambd=1: float. 
            Optional. Regularization parameter.
        K=None: np.array (shape=(n,n)). 
            Optional. Gram matrix if available.       
    """
    def train(self, Xtr, Ytr, n, lambd = 1, K=None):

        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(Xtr, Ytr, n, K)
        
        
        if self.verbose:
            print("Solving SVM opt problem...")


        P = matrix(self.K,tc='d')
        q = matrix(-Ytr,tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)),np.diag(Ytr.astype(float)),axis=0),tc='d')
        h = matrix(np.append(np.zeros(self.n),np.ones(self.n,dtype=float)/(2*lambd*self.n),axis=0),tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
        
        
        if self.verbose:
            print("end")
