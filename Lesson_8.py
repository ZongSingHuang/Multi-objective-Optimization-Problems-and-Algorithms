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
    
    # Applt the constraints
    Sy = 300e3
    delta_max = 0.005
    
    g = np.array([None, None])
    g[0] = -Sy + (32*p*l) / (np.pi*d**3) # ( (32*p*l) / (np.pi*d**3) ) - Sy <=0
    g[1] = -delta_max + (64*p*l**3)/(3*E*np.pi*d**4) # ( (64*p*l**3) / (3*E*np.pi*d**4) ) - delta_max <=0
    
    if any(g>0):
        return f1, f2, 0
    else:
        return f1, f2, 1

d = np.linspace(0.01, 0.05, 20)
l = np.linspace(0.2, 1, 20)
o = []
x = []

fig , ax = plt.subplots(1, 2, sharex=False, sharey=False)


for i in range(len(d)):
    for j in range(len(l)):
        temp_x = np.array([ d[i], l[j] ])
        temp_o = Cantilever(temp_x)
        
        if temp_o[2]==0:
            ax[0].scatter(temp_x[0], temp_x[1], color='black')
            ax[1].scatter(temp_o[0], temp_o[1], color='black')
        else:
            ax[0].scatter(temp_x[0], temp_x[1], color='red')
            ax[1].scatter(temp_o[0], temp_o[1], color='red')
o = np.array(o).reshape(-1, 2)
x = np.array(x)


ax[0].set_title('Search space')
ax[0].set_xlabel('d')
ax[0].set_ylabel('l')
ax[1].set_title('Objective space')
ax[1].set_xlabel('Weight')
ax[1].set_ylabel('Deflection')
fig.tight_layout()