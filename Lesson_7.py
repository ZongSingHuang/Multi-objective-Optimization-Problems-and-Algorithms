# -*- coding: utf-8 -*-
"""
Created on Mon Aug  9 11:05:52 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

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

d = np.linspace(0.01, 0.05, 20)
l = np.linspace(0.2, 1, 20)
o = []
x = []

for i in range(len(d)):
    for j in range(len(l)):
        x.append( np.array([ d[i], l[j] ]) )
        o.append( Cantilever(x[-1]) )
o = np.array(o).reshape(-1, 2)
x = np.array(x)

fig , ax = plt.subplots(1, 2, sharex=False, sharey=False)
ax[0].set_title('Search space')
ax[0].scatter(x[:, 0], x[:, 1])
ax[0].set_xlabel('d')
ax[0].set_ylabel('l')
ax[1].set_title('Objective space')
ax[1].scatter(o[:, 0], o[:, 1])
ax[1].set_xlabel('Weight')
ax[1].set_ylabel('Deflection')
fig.tight_layout()