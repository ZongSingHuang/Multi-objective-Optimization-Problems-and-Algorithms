# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:05:52 2021

@author: zongsing.huang
"""

import numpy as np

def ZTD1(x):
    # x in [0, 1]
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    n = x.shape[1]
    
    g = 1 + 9/(n-1) * np.sum(x[:, 1:], axis=1)
    f1 = x[:, 0]
    f2 = g * ( 1 - (x[:, 0]/g)**.5 )
    
    return f1, f2

def Cantilever(x):
    # d in [0.01, 0.05]
    # l in [0.2, 1]
    if x.ndim==1:
        x = x.reshape(1, -1)
        
    d = x[:, 0]
    l = x[:, 1]
    
    p = 1
    rho = 7800
    E = 207e6
    
    f1 = (rho * np.pi * d**2 * l) / 4
    f2 = (64 * p * l**3) / (3*E*np.pi*d**4)
    
    return f1, f2


x = np.array([.1, .2, 0, 0])
f1, f2 = ZTD1(x)

x = np.array([0.05, 1.0])
f1, f2 = Cantilever(x)