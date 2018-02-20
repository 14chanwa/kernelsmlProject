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
        return x.dot(y.transpose())


"""
    Gaussian_kernel
"""
class Gaussian_kernel:

    
    def __init__(self, gamma):
        self.gamma = gamma
    
    
    """
    Gaussian_kernel.evaluate
    
    Compute exp(- gamma * norm(x, y)^2).
    
    Parameters:
    x: np.array.
    y: np.array.
    
    Returns:
    res: float.
    """
    def evaluate(self, x, y):
        return np.exp(-self.gamma * np.sum(np.power(x - y, 2))) 


"""
    compute_matrix_K
    
    Compute K from data and kernel.
    
    Parameters:
    Xtr: np.array. Training data.
    kernel: Kernel. Kernel instance.
    
    Returns:
    K: np.array.
"""
def compute_matrix_K(Xtr, kernel, n):
    K = np.zeros([n, n], dtype=float)
    for i in range(n):
        K[i, i] = kernel.evaluate(Xtr[i], Xtr[i])
        for j in range(i):
            K[i, j] = kernel.evaluate(Xtr[i], Xtr[j])
            K[j, i] = K[i, j]
    return K


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
        if K is None:
            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
        else:
            self.K = K
        
        
        self.eta = np.sum(self.K) / np.power(self.n, 2)
        
        
        # Store centered kernel
        U = np.ones(self.K.shape) / self.n
        self.centered_K = (np.eye(self.n) - U).dot(self.K).dot(np.eye(self.n) - U)

    
    def get_centered_K(self):
        return self.centered_K
    
    
    """
        CenteredKernel.evaluate
        Compute the centered kernel value of x, y.
        
        Parameters:
        x: object.
        y: object.
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
    
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    def train(self, Xtr, Ytr, n, lambd=0.1, K=None, W=None):
        
        self.Xtr = Xtr
        self.n = n
        
        
        if K is None:
            if self.verbose:
                print("Build K...")
            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
            if self.verbose:
                print("end")
        else:
            self.K = K
        
        
        if self.center:
            if self.verbose:
                print("Center K...")
            self.centeredKernel = CenteredKernel(self.kernel)
            self.centeredKernel.train(self.Xtr, self.n, self.K)
            self.kernel = self.centeredKernel
            # replace K by centered kernel
            # it is not important to replace K since centering a centered kernel
            # leaves it unchanged.. in principle
            self.K = self.centeredKernel.get_centered_K() 
            if self.verbose:
                print("end")
        
        
        if W is None:
            W = np.eye(self.n,dtype=float)
            W_sqrt = W
        else:
            W_sqrt = np.sqrt(W)
        
#            K = np.zeros([self.n, self.n], dtype=float)
#            for i in range(self.n):
#                K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
#                for j in range(i):
#                    K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
#                    K[j, i] = K[i, j]
#        self.K = K
        
        tmp = np.linalg.inv(W_sqrt.dot(self.K).dot(W_sqrt) + lambd * self.n * np.eye(self.n))
        self.alpha = W_sqrt.dot(tmp).dot(W_sqrt).dot(Ytr)
    
    def predict(self, Xte, m):
        if self.verbose:
            print("Predict...")
        
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            tmp = np.zeros((self.n,))
            for j in range(self.n):
                tmp[j] = self.kernel.evaluate(self.Xtr[j], Xte[i])
            #tmp = np.multiply(self.alpha, tmp)
            #Yte[i] = np.sum(tmp)
            Yte[i] = tmp.dot(self.alpha)
        
        if self.verbose:
            print("end")
        
        return Yte


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
class LogisticRegression():
    
    # Maybe better: center=True as default?
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    def train(self, Xtr, Ytr, n, lambd = 1, K = None):
        
        
#        print("Centering the Gram matrix")
#        if self.center:
#            self.centeredKernel = CenteredKernel(self.kernel)
#            self.centeredKernel.train(Xtr, n,K)
#            self.kernel = self.centeredKernel
  
        self.n = n
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.lambd = lambd
#        self.K = K
        
#        print("Build the Gram Matrix")
#        if K is None:
#            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
#            self.K = np.zeros([self.n, self.n], dtype=float)
#            for i in range(self.n):
#                print(i)
#                self.K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
#                for j in range(i):
#                    self.K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
#                    self.K[j, i] = self.K[i, j]
        
        if K is None:
            if self.verbose:
                print("Build K...")
            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
            if self.verbose:
                print("end")
        else:
            self.K = K
        
        
        if self.center:
            if self.verbose:
                print("Center K...")
            self.centeredKernel = CenteredKernel(self.kernel)
            self.centeredKernel.train(self.Xtr, self.n, self.K)
            self.kernel = self.centeredKernel
            # replace K by centered kernel
            # it is not important to replace K since centering a centered kernel
            # leaves it unchanged.. in principle
            self.K = self.centeredKernel.get_centered_K() 
            if self.verbose:
                print("end")
        
        if self.verbose:
            print("Start IRLS")
        self.alpha = self.IRLS()
        
        
    def predict(self, Xte, m):
        if self.verbose:
            print("Predict...")
        
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            #print('Predict label %d' %i)
            tmp = np.zeros((self.n,))
            for j in range(self.n):
                tmp[j] = self.kernel.evaluate(self.Xtr[j], Xte[i])
            #tmp = np.multiply(self.alpha, tmp)
            #Yte[i] = np.sum(tmp)
            Yte[i] = tmp.dot(self.alpha)
        
        if self.verbose:
            print("end")
        return Yte

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
class SVM():
    
    def __init__(self, kernel=None, center=False, verbose=True):
        if kernel is None:
            self.kernel = Linear_kernel()
        else:
            self.kernel = kernel
        self.center = center
        
        self.verbose = verbose
    
    def train(self, Xtr, Ytr, n, lambd = 1, K=None):
        
#        if self.center:
#            self.centeredKernel = CenteredKernel(self.kernel)
#            self.centeredKernel.train(Xtr, n, K)
#            self.kernel = self.centeredKernel
            
        self.n = n
        self.Xtr = Xtr
#        self.K = K
#        if K is None:
#            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
        
        if K is None:
            if self.verbose:
                print("Build K...")
            self.K = compute_matrix_K(self.Xtr, self.kernel, self.n)
            if self.verbose:
                print("end")
        else:
            self.K = K
        
        
        if self.center:
            if self.verbose:
                print("Center K...")
            self.centeredKernel = CenteredKernel(self.kernel)
            self.centeredKernel.train(self.Xtr, self.n, self.K)
            self.kernel = self.centeredKernel
            # replace K by centered kernel
            # it is not important to replace K since centering a centered kernel
            # leaves it unchanged.. in principle
            self.K = self.centeredKernel.get_centered_K() 
            if self.verbose:
                print("end")
            
#            self.K = np.zeros([self.n, self.n], dtype=float)
#            for i in range(self.n):
#                self.K[i, i] = self.kernel.evaluate(Xtr[i], Xtr[i])
#                for j in range(i):
#                    self.K[i, j] = self.kernel.evaluate(Xtr[i], Xtr[j])
#                    self.K[j, i] = self.K[i, j]

        P = matrix(self.K,tc='d')
        q = matrix(-Ytr,tc='d')
        G = matrix(np.append(np.diag(-Ytr.astype(float)),np.diag(Ytr.astype(float)),axis=0),tc='d')
        h = matrix(np.append(np.zeros(self.n),np.ones(self.n,dtype=float)/(2*lambd*self.n),axis=0),tc='d')
        solvers.options['show_progress'] = False
        self.alpha = np.array(solvers.qp(P, q, G, h)['x'])
    
    def predict(self, Xte, m):
        if self.verbose:
            print("Predict...")
        
        Yte = np.zeros((m,), dtype=float)
        for i in range(m):
            tmp = np.zeros((self.n,))
            for j in range(self.n):
                tmp[j] = self.kernel.evaluate(self.Xtr[j], Xte[i])
            #tmp = np.multiply(self.alpha, tmp)
            Yte[i] = tmp.dot(self.alpha)
            #Yte[i] = np.sum(tmp)
        
        if self.verbose:
            print("end")
        
        return Yte

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
