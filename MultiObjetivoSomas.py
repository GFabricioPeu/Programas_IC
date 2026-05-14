import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def somas_ponderadas(w1, w2, restricoes, x0=[3.0, 3.0]):
    phi = lambda x: w1 * x[0] + w2 * x[1]   
    res = minimize(phi, x0, constraints=restricoes, method='SLSQP')
    return res.x if res.success else None

#Problema convexo (funciona)
npp1 = 20
restricoes_33 = [
    {'type': 'ineq', 'fun': lambda x: 1 - (x[0]-3)**2 - (x[1]-3)**2} # Exige que seja > = 0
]

PF33 = []
for i in range(0, npp1+1):
    w1 = i / npp1        
    w2 = 1 - w1         
    pt = somas_ponderadas(w1, w2, restricoes_33)
    if pt is not None:
        PF33.append(pt)
PF33 = np.array(PF33)

#Problema não convexo (não funciona)
npp2 = 30
restricoes_34 = [
    {'type': 'ineq', 'fun': lambda x: 5*(x[1]-1) + (x[0]-3)**3 - 2},
    {'type': 'ineq', 'fun': lambda x: 4.5 - x[0]},
    {'type': 'ineq', 'fun': lambda x: 3.0 - x[1]},
    # Exige que seja > = 0
]

PF34 = []
for i in range(1, npp2+1):
    w1 = i / npp2
    w2 = 1 - w1
    pt = somas_ponderadas(w1, w2, restricoes_34, x0=[3.5, 2.0])
    if pt is not None:
        PF34.append(pt)
PF34 = np.array(PF34)

#Problema Linear (não funciona)
npp3 = 30
restricoes_35 = [
    {'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},
    {'type': 'ineq', 'fun': lambda x: 1 - x[0]},
    {'type': 'ineq', 'fun': lambda x: 1 - x[1]},
]

PF35 = []
for i in range(1, npp3+1):
    w1 = i / npp3
    w2 = 1 - w1
    pt = somas_ponderadas(w1, w2, restricoes_35, x0=[0.5, 0.5])
    if pt is not None:
        PF35.append(pt)
PF35 = np.array(PF35)

#Plot
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

axs[0].scatter(PF33[:,0], PF33[:,1])
axs[0].set_title("Ex 3.3 — Convexa ")
axs[0].set_xlabel("f1"); axs[0].set_ylabel("f2")

axs[1].scatter(PF34[:,0], PF34[:,1], color='red')
axs[1].set_title("Ex 3.4 — Não Convexa")
axs[1].set_xlabel("f1"); axs[1].set_ylabel("f2")

axs[2].scatter(PF35[:,0], PF35[:,1], color='green')
axs[2].set_title("Ex 3.5 — Linear (só 3 pontos!)")
axs[2].set_xlabel("f1"); axs[2].set_ylabel("f2")

plt.tight_layout()
plt.savefig("somas_ponderadas.png", dpi=150)
plt.show()