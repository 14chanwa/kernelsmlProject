# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt


#%%

"""
    Linear_kernel
"""
class Linear_kernel:
    """
    Linear_kernel.evaluate
    Compute x.dot(y)
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    def evaluate(self, x, y):
        
        return x.dot(y)


"""
    Gaussian_kernel
"""
class Gaussian_kernel:
    """
    Gaussian_kernel.evaluate
    
    Compute exp(- gamma * norm(x, y)^2).
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    
    def __init__(self, gamma):
        self.gamma = gamma
    
    def evaluate(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y)**2) 


#%%

"""
    CenteredKernel
    Class that implements computing centered kernel values.
"""
class CenteredKernel():
    
    def __init__(self, kernel):
        self.kernel = kernel
    
    """
        CenteredKernel.train
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
        CenteredKernel.evaluate
        Compute the centered kernel value of x, y.
        
        Parameters:
        x: object.
        y: object.
    """
    def evaluate(self, x, y):
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

class RidgeRegression():
    
    def __init__(self, kernel=Linear_kernel(), center=False):
        self.kernel = kernel
        self.center = center
    
    def train(self, Xtr, Ytr, n, lambd=0.1, K=None, W=None):
        
        if self.center:
            self.centeredKernel = CenteredKernel(self.kernel)
            self.centeredKernel.train(Xtr, n)
            self.kernel = self.centeredKernel
        
        self.Xtr = Xtr
        self.n = n
        if W is None:
            W = np.eye(self.n,dtype=float)

        if K is None:
            K = np.zeros([self.n, self.n], dtype=float)
            for i in range(self.n):
                K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
                for j in range(i):
                    K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
                    K[j, i] = K[i, j]
        
        W_sqrt = np.sqrt(W)
        tmp = np.linalg.inv(W_sqrt.dot(K).dot(W_sqrt) + lambd * self.n * np.eye(self.n))
        self.alpha = W_sqrt.dot(tmp).dot(W_sqrt).dot(Ytr)
    
    def predict(self, Xte):
        m = Xte.shape[0]
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            tmp = np.zeros((self.n,))
            for j in range(self.n):
                tmp[j] = self.kernel.evaluate(self.Xtr[j], Xte[i])
            tmp = np.multiply(self.alpha, tmp)
            Yte[i] = np.sum(tmp)
        return Yte


# Test
print("Test on 2 vectors")
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
n = Xtr.shape[0]
ridge_regression = RidgeRegression(center=False)
ridge_regression.train(Xtr, Ytr, n, 0)
print(ridge_regression.predict(np.array([1, 2]).reshape((1, 2))))
print(ridge_regression.predict(np.array([1, 2.5]).reshape((1, 2))))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')
res = np.zeros(xv.shape)
for i in range(xv.shape[0]):
    for j in range(xv.shape[1]):
        res[i, j] = ridge_regression.predict(np.array([xv[i, j], yv[i, j]]).reshape((1, 2)))

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.show()

# Test
print("Test on 3 vectors + centered")
Xtr = np.array([[1, 1], [1, 3], [2, 1]]).reshape((3, 2))
Ytr = np.array([1, -1, 1]).reshape((3,))
n = Xtr.shape[0]
ridge_regression = RidgeRegression(center=True)
ridge_regression.train(Xtr, Ytr, n, 0)
print(ridge_regression.predict(np.array([1, 2]).reshape((1, 2))))
print(ridge_regression.predict(np.array([1, 2.5]).reshape((1, 2))))

xv, yv = np.meshgrid(np.arange(0, 5, 0.1), np.arange(0, 5, 0.1), sparse=False, indexing='xy')
res = np.zeros(xv.shape)
for i in range(xv.shape[0]):
    for j in range(xv.shape[1]):
        res[i, j] = ridge_regression.predict(np.array([xv[i, j], yv[i, j]]).reshape((1, 2)))

plt.axis('equal')
plt.scatter(xv, yv, c=res, s=100)
plt.colorbar()
plt.scatter(Xtr[:,0], Xtr[:, 1], color='red')
plt.show()

#%%
        
      
class LogisticRegression():
    
    def __init__(self, kernel=Linear_kernel(), center=False):
        self.kernel = kernel
    
    def train(self, Xtr, Ytr, n, lambd = 1, K = None):
        self.n = n
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.lambd = lambd
        if K is None:
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
            tmp = np.zeros((self.n,))
            for j in range(self.n):
                tmp[j] = self.kernel.evaluate(self.Xtr[j], Xte[i])
            tmp = np.multiply(self.alpha, tmp)
            Yte[i] = np.sum(tmp)
        return Yte

    def IRLS(self, K, precision = 1e-12, max_iter = 1000):
        self.K = K
        W = np.eye(self.n,dtype = float)

        alpha = np.ones(self.n,dtype = float)
        P = np.eye(self.n, dtype = float)
        
        for i in range(self.n):
            P[i,i] = -self.sigmoid(-self.Ytr[i]*self.K[i,].dot(alpha))
        
        z = self.K.dot(alpha) - np.linalg.inv(W).dot(P).dot(self.Ytr)
        ridge_regression = RidgeRegression()
        err = 1e12
        old_err = 0
        iter = 0
        while (iter < max_iter) & (np.abs(err-old_err) > precision):
            old_err = err
            ridge_regression.train(Xtr=self.Xtr, Ytr=z, lambd = self.lambd, n=self.n, K=self.K, W=W)
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
        J = 0.5*self.lambd*alpha.transpose().dot(self.K.dot(alpha))
        for i in range(self.n):
            J += self.log_loss(self.Ytr[i]*self.K[i,].dot(alpha)) / self.n
        return J
        
# Test
print("Test on 2 vectors")
n = 2
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
log_regression = LogisticRegression()
log_regression.train(Xtr, Ytr, n, 10)
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
    
