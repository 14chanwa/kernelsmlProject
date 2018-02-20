# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 02:02:07 2018

@author: Quentin
"""

import numpy as np


def generate_submission_file(classifier0, classifier1, classifier2, use_bow=True):
    if use_bow:
        Xte0 = np.genfromtxt('./data/Xte0_mat50.csv', delimiter=' ')
        Xte1 = np.genfromtxt('./data/Xte1_mat50.csv', delimiter=' ')
        Xte2 = np.genfromtxt('./data/Xte2_mat50.csv', delimiter=' ')
    else:
        Xte0 = np.genfromtxt('./data/Xte0.csv', delimiter=' ')
        Xte1 = np.genfromtxt('./data/Xte1.csv', delimiter=' ')
        Xte2 = np.genfromtxt('./data/Xte2.csv', delimiter=' ')
    
    # Generate labels
    Yte0 = np.array(np.sign(classifier0.predict(Xte0, Xte0.shape[0])), dtype=int)
    Yte1 = np.array(np.sign(classifier1.predict(Xte1, Xte1.shape[0])), dtype=int)
    Yte2 = np.array(np.sign(classifier2.predict(Xte2, Xte2.shape[0])), dtype=int)

    # Map {-1, 1} back to {0, 1}
    Yte0[Yte0 == -1] = 0
    Yte1[Yte1 == -1] = 0
    Yte2[Yte2 == -1] = 0
    
    f = open('results.csv', 'w')
    f.write("Id,Bound\n")
    count = 0
    for i in range(len(Yte0)):
        f.write(str(count)+","+str(Yte0[i])+"\n")
        count += 1
    for i in range(len(Yte1)):
        f.write(str(count)+","+str(Yte1[i])+"\n")
        count += 1
    for i in range(len(Yte2)):
        f.write(str(count)+","+str(Yte2[i])+"\n")
        count += 1
    f.close()