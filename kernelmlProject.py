# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:32:01 2018

@author: Quentin
"""

import numpy as np


def linear_kernel(x, y):
    return x.dot(y)


class CentreringDataClass():
    
    def __init__(self, kernel):
        self.kernel = kernel
    
    def train(self, Xtr):
        self.Xtr = Xtr
    
    def center(self, x):
        return 1 #TODO


class LinearRegression():
    
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
    
