# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np
import matplotlib.pyplot as plt

from kernelsmlProject.kernels import *
from kernelsmlProject.algorithms import *


# Test
print("Test: try to center 2 vectors")
n = 2
X = np.array([[0, 2], [0, 0]], dtype=float)
centeredKernel = CenteredKernel(LinearKernel())
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

### RIDGE REGRESSION
        

# Test
print("Test on 2 vectors")
Xtr = np.array([[1, 1], [1, 3]]).reshape((2, 2))
Ytr = np.array([1, -1]).reshape((2,))
n = Xtr.shape[0]
ridge_regression = RidgeRegression(LinearKernel(), center=False)
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

### LOGISTIC REGRESSION
        
    
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

### SVM
    

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
