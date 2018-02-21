# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt


#%%

class Kernel:
    
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
            K[i, i] = self.evaluate(Xtr[i], Xtr[i])
            for j in range(i):
                K[i, j] = self.evaluate(Xtr[i], Xtr[j])
                K[j, i] = K[i, j]
        return K


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



# Test
print("Test: try to center 2 vectors")
n = 2
X = np.array([[0, 2], [0, 0]], dtype=float)
centeredKernel = CenteredKernel(Linear_kernel())
centeredKernel.train(X, n)
print(centeredKernel.evaluate(X[0,:], X[1,:]))
print(centeredKernel.evaluate(np.array([1, 1]), X[1,:]))

print("Test: try to center gaussian kernel of mean 1")
n = 100
X = np.random.normal(1.0, 1.0, size=(100, 1))
print(np.mean(X))
centeredKernel.train(X, n)
Xc = np.zeros((100,))
for i in range(Xc.size):
    Xc[i] = centeredKernel.evaluate(np.array([1]).reshape((1, 1)), X[i])
print(np.mean(Xc))


#%%

"""
    RegressionInstance
    Base class for an instance of regression problem solver.
"""
class RegressionInstance():
    
    
    """
        RegressionInstance.predict
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
    def predict(self, Xte, m):
        if self.verbose:
            print("Predict...")
            
        K_t = self.kernel.get_test_K_evaluations(self.Xtr, self.n, Xte, m,\
                                                 self.verbose)
        Yte = K_t.dot(self.alpha.reshape((self.alpha.size,1))).reshape(-1)
        
        if self.verbose:
            print("end")
        
        return Yte


    """
        RegressionInstance.init_train
        Builds self.K or/and its centered version.
        
        Parameters
        ----------
        K: np.array (shape=(n,n)). 
            Kernel Gram matrix or None.
    """
    def init_train(self, K):
        
        if K is None:
            if self.verbose:
                print("Build K...")
            self.K = self.kernel.compute_matrix_K(self.Xtr, self.n)
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
            self.centeredKernel.train(self.Xtr, self.n, self.K)
            self.kernel = self.centeredKernel
            # replace K by centered kernel
            # it is not important to replace K since centering a centered kernel
            # leaves it unchanged.. in principle
            self.K = self.centeredKernel.get_centered_K() 
            if self.verbose:
                print("end")


"""
    RidgeRegression
    Implements ridge regression.
"""
class RidgeRegression(RegressionInstance):
    
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
        
        self.Xtr = Xtr
        self.n = n
        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(K)
        
        
        if W is None:
            W = np.eye(self.n,dtype=float)
            W_sqrt = W
        else:
            W_sqrt = np.sqrt(W)
        
        tmp = np.linalg.inv(W_sqrt.dot(self.K).dot(W_sqrt) + lambd * self.n * np.eye(self.n))
        self.alpha = W_sqrt.dot(tmp).dot(W_sqrt).dot(Ytr)


        """
            RidgeRegression.predict
            Inherited from RegressionInstance.
            predict(self, Xte, m)
        """
        

# Test
print("Test on 2 vectors")
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
n = Xtr.shape[0]
ridge_regression = RidgeRegression(center=False)
ridge_regression.train(Xtr, Ytr, n, 1e-3)
print(ridge_regression.predict(np.array([1, 2]).reshape((1, 2)), 1))
print(ridge_regression.predict(np.array([1, 2.5]).reshape((1, 2)), 1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')

print(np.concatenate((xv.reshape((xv.shape[0]*yv.shape[0],1)), yv.reshape((xv.shape[0]*yv.shape[0],1))), axis=1).shape)

m = xv.shape[0]*yv.shape[0]
res = ridge_regression.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(xv[np.abs(res)<1e-2],yv[np.abs(res)<1e-2], color= 'black')
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.show()

# Test
print("Test on 3 vectors + centered")
Xtr = np.array([[1, 1], [1, 3], [2, 1]]).reshape((3, 2))
Ytr = np.array([1, -1, 1]).reshape((3,))
n = Xtr.shape[0]
ridge_regression = RidgeRegression(center=True)
ridge_regression.train(Xtr, Ytr, n, 1e-4)
print(ridge_regression.predict(np.array([1, 2]).reshape((1, 2)), 1))
print(ridge_regression.predict(np.array([1, 2.5]).reshape((1, 2)), 1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')
m = xv.shape[0]*yv.shape[0]
res = ridge_regression.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.scatter(xv[np.abs(res)<1e-1],yv[np.abs(res)<1e-1], color= 'black')

plt.show()

#%%

"""
    LogisticRegression
    Implements logistic regression.
"""
class LogisticRegression(RegressionInstance):
    
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

  
        self.n = n
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.lambd = lambd

        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(K)
        
        
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
    
    
    """
        LogisticRegression.predict
        Inherited from RegressionInstance.
        predict(self, Xte, m)
    """
        
    
# Test
print("Test on 2 vectors")
n = 2
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))

#lambds = np.logspace(np.log10(0.01), np.log10(50),30)
#for lambd in lambds:
#    print("\n------ Lambda = %.3f ------" %lambd)
#    log_regression = LogisticRegression(center=True)
#    log_regression.train(Xtr, Ytr, n, lambd)
#    print(log_regression.predict(np.array([1, 2]).reshape((1, 2))))
#    print(log_regression.predict(np.array([1, 2.5]).reshape((1, 2))))
#    print(log_regression.alpha)
    
log_regression = LogisticRegression(center=True)
log_regression.train(Xtr, Ytr, n, 0.25)
print(log_regression.predict(np.array([1, 2]).reshape((1, 2)), 1))
print(log_regression.predict(np.array([1, 2.5]).reshape((1, 2)), 1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')

m = xv.shape[0]*yv.shape[0]
res = log_regression.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.scatter(xv[np.abs(res)<1e-2],yv[np.abs(res)<1e-2], color= 'black')
plt.show()

# Test
print("Test on 3 vectors + centered")
Xtr = np.array([[1, 1], [1, 3], [2, 1]]).reshape((3, 2))
Ytr = np.array([1, -1, 1]).reshape((3,))
n = Xtr.shape[0]
log_regression = LogisticRegression(center=True)
log_regression.train(Xtr, Ytr, n, 0.5)
print(log_regression.predict(np.array([1, 2]).reshape((1, 2)), 1))
print(log_regression.predict(np.array([1, 2.5]).reshape((1, 2)), 1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')

m = xv.shape[0]*yv.shape[0]
res = log_regression.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.scatter(xv[np.abs(res)<1e-2],yv[np.abs(res)<1e-2], color= 'black')
plt.show()


#%%
from cvxopt import matrix, solvers
class SVM(RegressionInstance):
    
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    def train(self, Xtr, Ytr, n, lambd = 1, K=None):
            
        self.n = n
        self.Xtr = Xtr
        
        # Call function from super that builds K if not built and initializes
        # self.K (centered or not)
        self.init_train(K)

        P = matrix(self.K,tc='d')
        q = matrix(-Ytr,tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)),np.diag(Ytr.astype(float)),axis=0),tc='d')
        h = matrix(np.append(np.zeros(self.n),np.ones(self.n,dtype=float)/(2*lambd*self.n),axis=0),tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
    
    
    """
        SVM.predict
        Inherited from RegressionInstance.
        predict(self, Xte, m)
    """
    
    

# Test
print("Test on 2 vectors")
n = 2
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
    
svm = SVM()
svm.train(Xtr, Ytr, n,0.05)
print(svm.predict(np.array([1, 2]).reshape((1, 2)),1))
print(svm.predict(np.array([1, 2.5]).reshape((1, 2)),1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')

m = xv.shape[0]*yv.shape[0]
res = svm.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.scatter(xv[np.abs(res)<1e-2],yv[np.abs(res)<1e-2], color= 'black')
plt.show()

# Test
print("Test on 3 vectors")
Xtr = np.array([[1, 1], [1, 3], [2, 1]]).reshape((3, 2))
Ytr = np.array([1, -1, 1]).reshape((3,))
n = Xtr.shape[0]
svm = SVM()
svm.train(Xtr, Ytr, n)
print(svm.predict(np.array([1, 2]).reshape((1, 2)),1))
print(svm.predict(np.array([1, 2.5]).reshape((1, 2)),1))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')

m = xv.shape[0]*yv.shape[0]
res = svm.predict(np.concatenate((xv.reshape((m,1)), yv.reshape((m,1))), axis=1), m).reshape(xv.shape)

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.scatter(xv[np.abs(res)<1e-2],yv[np.abs(res)<1e-2], color= 'black')
plt.show()    
