# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np
import matplotlib as plt


#%%

"""
    Linear_kernel
"""
class Linear_kernel:
    """
    Linear_kernel.evaluate
    
    Let x1, ... y1, ... be vectors of R^p.
    Suppose x, y are of form:
        x = [[x1], [x2], ...] (n lines)
        y = [[y1], [y2], ...] (m lines)
    Compute the matrix K such that K[i, j] = xi.dot(yj).
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    def evaluate(self, x, y):
        return x.dot(y.transpose())


"""
    Gaussian_kernel
"""
class Gaussian_kernel:
    """
    Gaussian_kernel.evaluate
    
    Let x1, ... y1, ... be vectors of R^p.
    Suppose x, y are of form:
        x = [[x1], [x2], ...] (n lines)
        y = [[y1], [y2], ...] (m lines)
    Compute the matrix K such that K[i, j] = exp(- gamma * norm(xi, yj)^2).
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    
    def __init__(self, gamma):
        self.gamma = gamma
    
    def evaluate(self, x, y):
        
        if y.ndim == 1:
            res = np.exp(- self.gamma * np.norm())
        
        n = x.shape[0]
        m = y.shape[0]
        res = np.zeros((n, m), dtype=float)
        for i in range(n):
            for j in range(m):
                res[i, j] = np.exp(-self.gamma * np.linalg.norm(x[i] - y[j])**2) 
        return res


#%%

"""
    CentreringInstance
    Class that implements computing centered kernel values.
"""
class CentreringInstance():
    
    def __init__(self, kernel):
        self.kernel = kernel
    
    """
        CentreringInstance.train
        Train the instance to center kernel values.
        
        Parameters:
        Xtr: list-like. Training data, of the form X = [[x1], ...].
        n: int. Number of lines of Xtr.
        K: np.array(shape=(n, n)). Optional: kernel matrix if available.    
    """
    def train(self, Xtr, n, K=None):
        self.Xtr = Xtr
        self.n = n
        if K != None:
            self.eta = np.sum(K.reshape(-1)) / self.n**2
        else:
            tmp = 0
            diag = 0
            for i in range(self.n):
                diag += self.kernel.evaluate(Xtr[i], Xtr[i])
                for j in range(i):
                    tmp += self.kernel.evaluate(Xtr[i], Xtr[j])
            self.eta = (2 * tmp + diag) / self.n**2
    
    
    """
        CentreringInstance.Kc
        Compute the centered kernel value of x, y.
        
        Parameters:
        x: ???
        y: ???
    """
    def Kc(self, x, y):
        res = self.kernel.evaluate(x, y)
        for k in range(self.n):
            res -= self.kernel.evaluate(self.Xtr[k], x) / self.n
            res -= self.kernel.evaluate(self.Xtr[k], y) / self.n
        res += self.eta
        return res


# Test
print("Test: try to center 2 vectors")
n = 2
X = np.array([[0, 2], [0, 0]], dtype=float)
centering_instance = CentreringInstance(Linear_kernel())
centering_instance.train(X, n)
print(centering_instance.Kc(X[0,:], X[1,:]))
print(centering_instance.Kc(np.array([1, 1]), X[1,:]))

print("Test: try to center gaussian kernel of mean 1")
n = 100
X = np.random.normal(1.0, 1.0, size=(100, 1))
print(np.mean(X))
centering_instance.train(X, n)
Xc = centering_instance.Kc(np.array([1]).reshape((1, 1)), X)
print(np.mean(Xc))

#%%

class RidgeRegression():
    
    def __init__(self, kernel=Linear_kernel(), center=False):
        self.kernel = kernel
    
    def train(self, Xtr, Ytr, n, lambd=1, K=None, W=None):
        self.Xtr = Xtr
        self.n = n
        if W == None:
            W = np.eye(self.n,dtype=float)

        if K == None:
            K = np.zeros([self.n, self.n], dtype=float)
            for i in range(self.n):
                K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
                for j in range(i):
                    K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
                    K[j, i] = K[i, j]
        print(K)
        W_sqrt = np.sqrt(W)
        self.alpha = W_sqrt.dot(np.linalg.inv(W_sqrt.dot(K.dot(W_sqrt)) + lambd * self.n * np.eye(self.n))).dot(W_sqrt.dot(Ytr))
    
    def predict(self, Xte):
        m = Xte.shape[0]
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            tmp = self.kernel.evaluate(self.Xtr, Xte[i])
            tmp = np.multiply(self.alpha, tmp)
            Yte[i] = np.sum(tmp)
        return Yte


# Test
print("Test on 2 vectors")
n = 2
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
ridge_regression = RidgeRegression()
ridge_regression.train(Xtr, Ytr, n, 0)
print(ridge_regression.predict(np.array([1, 2]).reshape((1, 2))))
print(ridge_regression.predict(np.array([1, 2.5]).reshape((1, 2))))

#%%
        
      
class LogisticRegression():
    
    def __init__(self, kernel=Linear_kernel(), center=False):
        self.kernel = kernel
    
    def train(self, Xtr, Ytr, n, lambd = 1, K = None):
        # TODO
        self.n = n
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.lambd = lambd
        if K == None:
            K = np.zeros([self.n, self.n], dtype=float)
            for i in range(self.n):
                K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
                for j in range(i):
                    K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
                    K[j, i] = K[i, j]

        self.alpha = self.IRLS(K)
    
    def predict(self, Xte):
        m = Xte.shape[0]
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            tmp = self.kernel.evaluate(self.Xtr, Xte[i])
            tmp = np.multiply(self.alpha, tmp)
            Yte[i] = np.sum(tmp)
        return Yte

    def IRLS(self, K, precision = 1e-12, max_iter = 1000):
        self.K = K
        W = np.eye(self.n,dtype = float)

        alpha = np.ones(self.n,dtype = float)
        P = np.eye(self.n, dtype = float)
        
        for i in range(self.n):
            P[i,i] = -self.log_loss(-self.Ytr[i]*self.K[i,].dot(alpha))
        
        z = self.K.dot(alpha) - np.linalg.inv(W).dot(P.dot(self.Ytr))
        ridge_regression = RidgeRegression()
        err = 1e12
        old_err = 0
        iter = 0
        while (iter < max_iter) & (np.abs(err-old_err) > precision):
            old_err = err
            ridge_regression.train(Xtr=self.Xtr, Ytr = z, n= self.n, K = self.K, W = W)
            alpha = ridge_regression.alpha
            print(iter)
            print(alpha)
            m = self.K.dot(alpha)
            for i in range(self.n):
                P[i,i] = -self.sigmoid(-self.Ytr[i]*m[i])
                W[i,i] = self.sigmoid(m[i])*self.sigmoid(-m[i])
                z[i] = m[i] + self.Ytr[i]/self.sigmoid(-self.Ytr[i]*m[i])
            err = self.distortion(alpha)
            iter += 1
        return alpha

        
    def log_loss(self,u):
        return np.log(1+np.exp(-u))
        
    def sigmoid(self,u):
        return 1/(1+np.exp(-u))
        
    def distortion(self, alpha):
        J = 0.5*self.lambd*alpha.dot(self.K.dot(alpha))
        for i in range(self.n):
            J += self.log_loss(self.Ytr[i]*self.K[i,].dot(alpha))
        return J
        
# Test
print("Test on 2 vectors")
n = 2
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
log_regression = LogisticRegression()
log_regression.train(Xtr, Ytr, n, 0)
print(log_regression.predict(np.array([1, 2]).reshape((1, 2))))
print(log_regression.predict(np.array([1, 2.5]).reshape((1, 2))))

#%%
class SVM():
    
    def __init__(self, kernel, center=False):
        self.kernel = kernel
    
    def train(self, Xtr, Ytr):
        # TODO
        self.Xtr = Xtr
        # self.alpha = ...
    
    def classify(self, Xte):
        # TODO
        #Yte = 
        #return Yte
        1
    
