# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 15:18:09 2021

@author: zongsing.huang
"""

import numpy as np

def dominates(x, y):
    # method 1
    o = int(all(x<=y) and any(x<y))
    
    # method 2
    # L = len(x)
    # lessThanOrEqualTo = np.zeros(L) - 1
    # lessThan = np.zeros_like(lessThanOrEqualTo) - 1
    
    # for i in range(L):
    #     if x[i]<=y[i]: # first condition of Pareto optiamlity
    #         lessThanOrEqualTo[i] = 1
    #     else:
    #         lessThanOrEqualTo[i] = 0
        
    #     if x[i]<y[i]: # second condition of Pareto optiamlity
    #         lessThan[i] = 1
    #     else:
    #         lessThan[i] = 0
    
    # if np.sum(lessThanOrEqualTo)==L and np.sum(lessThan)>=1:
    #     o = 1
    # else:
    #     o = 0
    
    return o
    
x = np.array([0, 0, 1, 1, 2])
y = np.array([0, 0, 1, 1, 3])
print(dominates(x, y))