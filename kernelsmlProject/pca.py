# -*- coding: utf-8 -*-
"""
Created on Thu Mar 01 23:22:30 2018


Implements kernel PCA and visualization functions.


@author: Quentin
"""

from kernelsmlProject.kernels import *
import numpy as np
import scipy.sparse.linalg as sparselinalg
import matplotlib.pyplot as plt


########################################################################
### PCA                                                         
########################################################################


class PCA():
    """
        PCA
        
        Attributes
        ----------
        self.kernel: Kernel.
        self.X_tr: list(object).
        self.n: int.
            Size of the training set.
        self.p: int.
            Number of principal components kept.
        self.K_tr: np.array((n, n)).
            Training Gram matrix.
        self.gamma: np.array((p, n)).
            Computed principal components.
        self.K_te: np.array((m, n)).
            Last used test Gram matrix.
    """
    
    def __init__(self, kernel):
        if not isinstance(kernel, CenteredKernel):
            #raise Exception("You must provide the PCA a CenteredKernel")
            self.kernel = CenteredKernel(kernel)
        else:
            self.kernel = kernel
    
    def compute_principal_components(self, Xtr, n, Ktr=None, p=None, verbose=True):
        """
            PCA.compute_principal_components
            Get the p principal components Gram matrix in the RHKS (by
            decreasing magnitude).

            Parameters
            ----------
            Xtr: list(object).
                Training data.
            n: int.
                Length of Xtr.
            Ktr: np.array((n,?)).
                Training K if available.
            p: int.
                Number of principal components to compute.
            verbose: bool.
        """
        
        self.X_tr = Xtr
        self.n = n
        if Ktr is not None:
            self.K_tr = Ktr
        else:
            self.K_tr = self.kernel.compute_K_train(Xtr, n, verbose=verbose)
        
        if p is None:
            p = n
        
        if p > n-1:
            raise Exception("p must be < n")
        
        # Compute the first p eigenmodes of the real sym square matrix K_tr
        delta, u = sparselinalg.eigsh(self.K_tr, k=p, which="LM")
        
        # Normalize
        self.gamma = np.zeros((p, n))
        for i in range(p):
            self.gamma[i, :] = u[:, i] / np.sqrt(delta[i])
    
    def get_projected_train(self):
        """
            PCA.get_projected_train
            Get the projections of the training set on the p principal
            components in the RKHS.

            Returns
            ----------
            res: np.array (shape=(n,p)).
                Projection of the training set on the p principal components.
        """
        
        return self.K_tr.dot(self.gamma.T)
        
    def get_projections(self, Xte, m, Kte=None, verbose=True):
        """
            PCA.get_projections
            Get the projections of the provided test set on the p principal
            components in the RKHS.

            Returns
            ----------
            res: np.array (shape=(n,p)).
                Projection of the test set on the p principal components.
        """
        
        # Compute new Gram matrix
        self.K_te = self.kernel.compute_K_test(self.X_tr, self.n, Xte, m, verbose=verbose)
        
        return self.K_te.dot(self.gamma.T)


########################################################################
### visualize_training_data_in_RKHS                                                         
########################################################################


def visualize_training_data_in_RKHS(kernel, Xtr, Ytr, n):
    """
        visualize_training_data_in_RKHS
        Projects the training data on the 6 main principal components in
        the RKHS and plot, with different colors for each label in {-1, 1}.
    """
    pca = PCA(kernel)
    pca.compute_principal_components(Xtr, n, p=6, verbose=True)
    projected_training_data = pca.get_projected_train()
    
    nb_pts_by_category = 250
    # Plot at most nb_pts_by_category pts of each category in each plot
    # Find indices == 1 and sample nb_pts_by_category indices among them
    ind_1 = np.argwhere(Ytr==1)
    ind_1_bis = np.random.randint(0, ind_1.size, min(nb_pts_by_category, ind_1.size))
    ind_1 = ind_1[ind_1_bis]
    # Find indices == -1 and sample 100 indices among them
    ind_m1 = np.argwhere(Ytr==-1)
    ind_m1_bis = np.random.randint(0, ind_m1.size, min(nb_pts_by_category, ind_m1.size))
    ind_m1 = ind_m1[ind_m1_bis]
    
    fig, ax = plt.subplots(1, 3)
    
    # gamma_1 fun gamma_0
    # -1
    ax[0].scatter(
        projected_training_data[ind_m1, 0], 
        projected_training_data[ind_m1, 1], 
        color="red",
        s=1
        )
    # 1
    ax[0].scatter(
        projected_training_data[ind_1, 0], 
        projected_training_data[ind_1, 1], 
        color="blue",
        s=1
        )
    ax[0].set_xlabel("$\gamma_0$")
    ax[0].set_ylabel("$\gamma_1$")
    
    # gamma_3 fun gamma_2
    # -1
    ax[1].scatter(
        projected_training_data[ind_m1, 2], 
        projected_training_data[ind_m1, 3], 
        color="red",
        s=1
        )
    # 1
    ax[1].scatter(
        projected_training_data[ind_1, 2], 
        projected_training_data[ind_1, 3], 
        color="blue",
        s=1
        )
    ax[1].set_xlabel("$\gamma_2$")
    ax[1].set_ylabel("$\gamma_3$")
    
    # gamma_5 fun gamma_4
    # -1
    ax[2].scatter(
        projected_training_data[ind_m1, 4], 
        projected_training_data[ind_m1, 5], 
        color="red",
        s=1
        )
    ax[2].set_xlabel("$\gamma_4$")
    ax[2].set_ylabel("$\gamma_5$")
    # 1
    ax[2].scatter(
        projected_training_data[ind_1, 4], 
        projected_training_data[ind_1, 5], 
        color="blue",
        s=1
        )
    ax[2].set_xlabel("$\gamma_4$")
    ax[2].set_ylabel("$\gamma_5$")
    
    plt.tight_layout()
    fig.suptitle("Principal components plot")
    fig.subplots_adjust(top=0.88)
    plt.show(block=True)
