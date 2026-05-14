import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# TRUQUE DA VARIÁVEL AUXILIAR t
#
# (T): min  max{ w1*(f1(x)-u1*), w2*(f2(x)-u2*) }
#
# Equivale a:
#   min   t
#   s.a   t - w1*(x1 - u1*) >= 0
#         t - w2*(x2 - u2*) >= 0
#         x ∈ X
#
# Variável de decisão: z = [x1, x2, t]
# ─────────────────────────────────────────────

def tchebychev(w1, w2, u_star, restricoes_X, x0):
    u1, u2 = u_star

    # objetivo: minimizar t = z[2]
    phi = lambda z: z[2]

    # restrições de Tchebychev
    # IMPORTANTE: w1 e w2 capturados com argumento default
    # para evitar problema de closure no loop
    rest_t = [
        {'type': 'ineq', 'fun': lambda z, w1=w1, u1=u1: z[2] - w1*(z[0] - u1)},
        {'type': 'ineq', 'fun': lambda z, w2=w2, u2=u2: z[2] - w2*(z[1] - u2)},
    ]

    # restrições originais de X reescritas em z
    rest_X = [{'type': r['type'],
               'fun': lambda z, f=r['fun']: f([z[0], z[1]])}
              for r in restricoes_X]

    res = minimize(phi, x0,
                   constraints=rest_t + rest_X,
                   method='SLSQP',
                   options={'ftol': 1e-9, 'maxiter': 2000})

    return res.x[:2] if res.success else None



# EXEMPLO 3.10 — Frente Convexa
# min {x1, x2}  s.a  (x1-3)^2 + (x2-3)^2 <= 1

npp = 20
c310 = [{'type': 'ineq', 'fun': lambda x: 1 - (x[0]-3)**2 - (x[1]-3)**2}]

PF310 = []


for i in range(1, npp+1):
    w1 = i / npp
    w2 = 1 - w1
    pt = tchebychev(w1, w2, (2.0, 2.0), c310, x0=[3.0, 3.0, 1.0])
    if pt is not None:
        PF310.append(pt)
PF310 = np.array(PF310)


# EXEMPLO 3.11 — Frente Nao Convexa
# min {x1, x2}  s.a  5(x2-1)+(x1-3)^3 >= 2
#                    x1 <= 4.5,  x2 <= 3

npp311 = 30
c311 = [
    {'type': 'ineq', 'fun': lambda x: 5*(x[1]-1) + (x[0]-3)**3 - 2},
    {'type': 'ineq', 'fun': lambda x: 4.5 - x[0]},
    {'type': 'ineq', 'fun': lambda x: 3.0 - x[1]},
]

PF311 = []
for i in range(1, npp311+1):
    w1 = i / npp311
    w2 = 1 - w1
    pt = tchebychev(w1, w2, (0.5, 0.5), c311, x0=[3.5, 2.0, 1.0])
    if pt is not None:
        PF311.append(pt)
PF311 = np.array(PF311)


# EXEMPLO 3.12 — Influencia do u* (Figura 22)
# min {x1, x2}  s.a  (x1-3)^2+(x2-3)^2 >= 1
#                    3 <= x1 <= 4,  3 <= x2 <= 4

npp312 = 30
c312 = [
    {'type': 'ineq', 'fun': lambda x: (x[0]-3)**2 + (x[1]-3)**2 - 1},
    {'type': 'ineq', 'fun': lambda x: 4 - x[0]},
    {'type': 'ineq', 'fun': lambda x: 4 - x[1]},
    {'type': 'ineq', 'fun': lambda x: x[0] - 3},
    {'type': 'ineq', 'fun': lambda x: x[1] - 3},
]

# Figura 22(a): u* = f* = (3,3) -> bem distribuida
PF312a = []
for i in range(1, npp312+1):
    w1 = i / npp312
    w2 = 1 - w1
    pt = tchebychev(w1, w2, (3.0, 3.0), c312, x0=[3.5, 3.5, 1.0])
    if pt is not None:
        PF312a.append(pt)
PF312a = np.array(PF312a)

# Figura 22(b): u* = (1,1) -> aglomerada
PF312b = []
for i in range(1, npp312+1):
    w1 = i / npp312
    w2 = 1 - w1
    pt = tchebychev(w1, w2, (1.0, 1.0), c312, x0=[3.5, 3.5, 1.0])
    if pt is not None:
        PF312b.append(pt)
PF312b = np.array(PF312b)


# Plot

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0,0].scatter(PF310[:,0], PF310[:,1], color='steelblue', s=60)
axs[0,0].set_title("Ex 3.10 - Convexa (funciona)")
axs[0,0].set_xlabel("f1"); axs[0,0].set_ylabel("f2"); axs[0,0].grid(True)

axs[0,1].scatter(PF311[:,0], PF311[:,1], color='tomato', s=60)
axs[0,1].set_title("Ex 3.11 - Nao Convexa (funciona!)")
axs[0,1].set_xlabel("f1"); axs[0,1].set_ylabel("f2"); axs[0,1].grid(True)

axs[1,0].scatter(PF312a[:,0], PF312a[:,1], color='seagreen', s=60)
axs[1,0].set_title("Ex 3.12(a) - u*=f*=(3,3) bem distribuida")
axs[1,0].set_xlabel("f1"); axs[1,0].set_ylabel("f2"); axs[1,0].grid(True)

axs[1,1].scatter(PF312b[:,0], PF312b[:,1], color='darkorange', s=60)
axs[1,1].set_title("Ex 3.12(b) - u*=(1,1) aglomerada")
axs[1,1].set_xlabel("f1"); axs[1,1].set_ylabel("f2"); axs[1,1].grid(True)

plt.suptitle("Tchebychev com variavel auxiliar t (SLSQP)", fontsize=15)
plt.tight_layout()
plt.savefig("tchebychev.png", dpi=150, bbox_inches='tight')
plt.show()