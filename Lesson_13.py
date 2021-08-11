# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 09:37:09 2021

@author: zongsing.huang
"""

import numpy as np
import matplotlib.pyplot as plt

def ZTD1(x, ZTD1_ideal):
    # x in [0, 1]
    if x.ndim==1:
        x = x.reshape(1, -1)
    
    n = x.shape[1]
    
    g = 1 + 9/(n-1) * np.sum(x[:, 1:], axis=1)
    F1 = x[:, 0]
    F2 = g * ( 1 - (x[:, 0]/g)**.5 )
    
    F1_with_F2 = np.vstack([F1, F2]).T
    F = np.linalg.norm(F1_with_F2-ZTD1_ideal, ord=2, axis=1)
    
    return F1, F2, F

#%% 參數設定
ZTD1_ideal = np.array([0, 0])
D = 10
P = 30
G = 500
w_max = 0.9
w_min = 0.2
c1 = 2
c2 = 2
k = 0.2
ub = np.ones(D)
lb = np.zeros(D)
v_max = (ub-lb) * k
v_min = -1*v_max

#%% 初始化
X = np.random.uniform(low=lb, high=ub, size=[P, D])
V = np.zeros([P, D])
pbest_X = np.zeros_like(X)
pbest_F = np.zeros(P) + np.inf
gbest_X = np.zeros(D)
gbest_F = np.inf
gbest_F1 = []
gbest_F2 = []
ub = ub*np.ones_like(X)
lb = lb*np.ones_like(X)
v_max = v_max*np.ones_like(X)
v_min = v_min*np.ones_like(X)
loss_curve = np.zeros(G)

#%% 迭代
for g in range(G):
    
    # 適應值計算
    F1, F2, F = ZTD1(X, ZTD1_ideal)
    
    # 更新pbest
    mask = F<pbest_F
    pbest_X[mask] = X[mask]
    pbest_F[mask] = F[mask]
    
    # 更新gbest
    if F.min()<gbest_F:
        min_idx = np.argmin(F)
        gbest_F = F[min_idx]
        gbest_F1.append(F1[min_idx])
        gbest_F2.append(F2[min_idx])
        gbest_X = X[min_idx]
    loss_curve[g] = gbest_F
    
    # 更新w
    w = w_max - g*(w_max-w_min)/G
    
    # 更新V
    r1 = np.random.uniform(size=[P, D])
    r2 = np.random.uniform(size=[P, D])
    V = w*V + c1*r1*(pbest_X-X) + c2*r2*(gbest_X-X)
    # 邊界處理
    mask1 = V>v_max
    mask2 = V<v_min
    V[mask1] = v_max[mask1]
    V[mask2] = v_min[mask2]
    
    # 更新X
    X = X + V
    # 邊界處理
    mask1 = X>ub
    mask2 = X<lb
    X[mask1] = ub[mask1]
    X[mask2] = lb[mask2]

#%% 作圖
plt.figure()
drawTruePF = np.loadtxt('ZDT1.txt')
plt.plot(drawTruePF[:, 0], drawTruePF[:, 1], label='Pareto set', color='grey')
plt.scatter(ZTD1_ideal[0], ZTD1_ideal[1], label='Ideal', color='red')
plt.scatter(gbest_F1, gbest_F2, label='PSO', color='blue')
[plt.text(gbest_F1[i]+0.01, gbest_F2[i]+0.01, i) for i in range(len(gbest_F1))]
plt.xlabel('f1')
plt.ylabel('f2')
plt.grid()
plt.legend()
plt.tight_layout()
