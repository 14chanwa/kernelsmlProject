# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np
import matplotlib as plt


#%%

"""
    linear_kernel
"""
class linear_kernel:
    """
    linear_kernel.evaluate
    
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
    def evaluate(x, y):
        return x.dot(y.transpose())


"""
    gaussian_kernel
"""
class gaussian_kernel:
    """
    gaussian_kernel.evaluate
    
    Let x1, ... y1, ... be vectors of R^p.
    Suppose x, y are of form:
        x = [[x1], [x2], ...] (n lines)
        y = [[y1], [y2], ...] (m lines)
    Compute the matrix K such that K[i, j] = - gamma * np.norm(xi, yj)^2.
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    
    def __init__(self, gamma):
        self.gamma = gamma
    
    def evaluate(x, y):
        return 1


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
centering_instance = CentreringInstance(linear_kernel)
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
    
    def __init__(self, kernel=linear_kernel, center=False):
        self.kernel = kernel
    
    def train(self, Xtr, Ytr, n, lambd=1, K=None):
        self.Xtr = Xtr
        self.n = n
        if K == None:
            K = np.zeros((self.n, self.n), dtype=float)
            for i in range(self.n):
                K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
                for j in range(i):
                    K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
                    K[j, i] = K[i, j]
        print(K)
        self.alpha = np.linalg.inv(K + lambd * self.n * np.eye(self.n)).dot(Ytr)
    
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
    
