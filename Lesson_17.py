# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:37:09 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt
import itertools

np.random.seed(42)

def selection(x, f):
    P = x.shape[0]
    p1_idx, p2_idx = np.random.choice(P, size=2, replace=False)
    
    p1 = x[p1_idx]
    p2 = x[p2_idx]
    p1_F = F[p1_idx]
    p2_F = F[p2_idx]
    
    return p1, p2, p1_F, p2_F

def crossover(p1, p2, pc, ub, lb):
    D = len(p1)
    beta = np.random.uniform(size=[D])
    
    c1 = beta*p1 + (1-beta)*p2
    c2 = beta*p2 + (1-beta)*p1
    
    mask1 = c1>ub
    mask2 = c1<lb
    c1[mask1] = ub[mask1]
    c1[mask2] = lb[mask2]
    mask1 = c2>ub
    mask2 = c2<lb
    c2[mask1] = ub[mask1]
    c2[mask2] = lb[mask2]
    
    r1 = np.random.uniform()
    if r1<=pc:
        c1 = c1
    else:
        c1 = p1
    r2 = np.random.uniform()
    if r2<=pc:
        c2 = c2
    else:
        c2 = p2
        
    return c1, c2

def mutation(c1, pm):
    D = len(c1)
    r = np.random.uniform(size=[D])
    
    mask = r<pm
    c1[mask] = np.random.uniform(low=lb[mask], high=ub[mask])
    
    return c1

def dominates(x, y):
    if x.ndim==2:
        x = x.flatten()
    if y.ndim==2:
        y = y.flatten()
    o = all(x<=y) and any(x<y)
    
    return o

def ZTD1(x):
    # x in [0, 1]
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    n = x.shape[1]
    
    g = 1 + 9/(n-1) * np.sum(x[:, 1:], axis=1)
    F1 = x[:, 0]
    F2 = g * ( 1 - (x[:, 0]/g)**.5 )
    
    return np.vstack([F1, F2]).T

#%% 參數設定
ZTD1_ideal = np.array([0, 0])
D = 2
P = 50
G = 100
pc = 0.95
pm = 0.5
er = 0.2
ub = np.ones(D)
lb = np.zeros(D)

#%% 初始化
X = np.random.uniform(low=lb, high=ub, size=[P, D])
F = ZTD1(X)
drawTruePF = np.loadtxt('ZDT1.txt')

#%% 迭代
for g in range(G):
    new_X = []
    new_F = []
    
    for k in range(P):
        # 選擇
        p1, p2, p1_F, P2_F = selection(X, F)
        
        # 交配
        c1, _ = crossover(p1, p2, pc, ub, lb)
        
        # 突變
        c1 = mutation(c1, pm)
        
        # 適應值計算
        c1_F = ZTD1(c1)
        
        # 子代
        if dominates(c1_F, p1_F)==True:
            new_X.append(c1)
            new_F.append(list(c1_F.flatten()))
        elif dominates(p1_F, c1_F)==True:
            new_X.append(p1)
            new_F.append(list(p1_F.flatten()))
        else:
            new_X.append(c1)
            new_F.append(list(c1_F.flatten()))
            new_X.append(p1)
            new_F.append(list(p1_F.flatten()))
    
    # 非支配解彼此間的競爭
    new_X = np.array(new_X)
    new_F = np.array(new_F)
    new_P = len(new_X)
    Dominated = np.zeros(new_P)
    similarPS = np.zeros_like(Dominated)
    
    for k, j in itertools.combinations(range(new_P), 2):
        if all(new_F[k]==new_F[j]):
            similarPS[k] = similarPS[k] + 1

        if dominates(new_F[k], new_F[j])==True:
            Dominated[j] = 1
        elif dominates(new_F[j], new_F[k])==True:
            Dominated[k] = 1
            
    idx = 0
    for k in range(new_P):
        if Dominated[k]==0 and similarPS[k]==0:
            X[idx] = new_X[k]
            F[idx] = new_F[k]
            idx = idx + 1

#%% 作圖
    plt.figure()
    plt.title(g)
    plt.plot(drawTruePF[:, 0], drawTruePF[:, 1], color='grey')
    for k in range(idx):
        plt.text(F[k, 0]+0.01, F[k, 1]+0.01, k)
        plt.scatter(F[k, 0], F[k, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.grid()
    plt.tight_layout()
    plt.show()