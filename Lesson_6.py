# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:05:52 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

def ZTD1(x):
    # x in [0, 1]
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    n = x.shape[1]
    
    g = 1 + 9/(n-1) * np.sum(x[:, 1:], axis=1)
    f1 = x[:, 0]
    f2 = g * ( 1 - (x[:, 0]/g)**.5 )
    
    return f1, f2

x1 = np.linspace(0, 1, 50)
x2 = np.linspace(0, 1, 50)
o = []
x = []

for i in range(len(x1)):
    for j in range(len(x2)):
        x.append( np.array([ x1[i], x2[j] ]) )
        o.append( ZTD1(x[-1]) )
o = np.array(o).reshape(-1, 2)
x = np.array(x)

fig , ax = plt.subplots(1, 2, sharex=False, sharey=False)
ax[0].set_title('Parameter space')
ax[0].scatter(x[:, 0], x[:, 1])
ax[0].set_xlabel('x1')
ax[0].set_ylabel('x2')
ax[1].set_title('Objective space')
ax[1].scatter(o[:, 0], o[:, 1])
ax[1].set_xlabel('o1')
ax[1].set_ylabel('o2')
fig.tight_layout()