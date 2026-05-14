import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def tchebychev_raios_principal(w1,w2,u_star,restricoes_X,x0):
    """
    Resolve o subproblema (TR) para um raio fixo (w1, w2).

    Variável auxiliar t = z[2] transforma o min-max em min linear:

        min   t
        s.a.  t  >=  w1 * (f1(x) - u1*)     [t domina o 1ºobjetivo]
              t  >=  w2 * (f2(x) - u2*)     [t domina o 2ºobjetivo]
              w2*(f2-u2*) >= w1*(f1-u1*)    [RESTRIÇÃO DO RAIO → garante que a solução fique 
                                            exatamente no raio de ângulo θ=arctan(w1/w2)]
            x ∈ X                         [restrições originais do problema]

    """
    u1,u2 = u_star
    
    phi = lambda z: z[2]
   
    rest_tr = [
        {'type': 'ineq', 'fun': lambda z, w1=w1, u1=u1: z[2] - w1*(z[0] - u1)},
        {'type': 'ineq', 'fun': lambda z, w2=w2, u2=u2: z[2] - w2*(z[1] - u2)},
        {'type': 'ineq', 'fun': lambda z, w1=w1, w2=w2, u1=u1, u2=u2: w2*(z[1] - u2) - w1*(z[0] - u1)}, #Restrição Raio
    ]

    rest_X = [{'type': r['type'],'fun': lambda z, f=r['fun']: 
               f([z[0], z[1]])}for r in restricoes_X]
    
    res = minimize(phi, x0,
                   constraints=rest_tr + rest_X,
                   options={'ftol': 1e-10, 'maxiter': 2000})
    return res.x[:2] if res.success else None
    

def espacamento_angulos (f1_star,f2_line,f1_line,f2_star,u_star,npp):
    '''
    Gera angulos igualmente espaçados entre theta_min e theta_max, medidos a partir do pt utópico u* até os extremos

    θ_min = arctan2( f2* - u2*,  f1' - u1* )   
    θ_max = arctan2( f2' - u2*,  f1* - u1* )   
    Δθ    = (θ_max - θ_min) / npp
    θ_k   = θ_min + k·Δθ,   k = 1, 2, ..., até npp

    '''
    u1, u2 = u_star
    theta_min = np.arctan2(f2_star - u2, f1_line  - u1)
    theta_max = np.arctan2(f2_line  - u2, f1_star  - u1)
    delta     = (theta_max - theta_min) / npp
    thetas    = [theta_min + k * delta for k in range(1, npp + 1)]
    return thetas

def limpeza_nao_paretos(pontos,M = 1e-8):
    '''
    Remove os pontos dominados não paretos

    Um ponto P^i é dominado por P^j se:
        f1(P^j) ≤ f1(P^i)  E  f2(P^j) ≤ f2(P^i)
        com pelo menos uma desigualdade ESTRITA.

    '''
    n = len(pontos)
    k = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j: continue
            if (pontos[j,0] <= pontos[i,0] + M and
                pontos[j,1] <= pontos[i,1] + M and
               (pontos[j,0] <  pontos[i,0] - M or
                pontos[j,1] <  pontos[i,1] - M)):
                k[i] = False; break
    return pontos[k], pontos[~k]

def encontra_minimo(fun, restricoes,x0):
    '''
    Resolve os problemas mono objetivos 
     Resolve min f(x) s.a. x ∈ X para achar os dois extremos da frente:
        P^0 = arg min f1(x)   → extremo esquerdo  (menor f1 possível)
        P^n= arg min f2(x)   → extremo inferior  (menor f2 possível)
    '''
    r = minimize(fun, x0[:2], constraints=restricoes,options={'ftol':1e-10,'maxiter':2000})

    return r.x if r.success else x0[:2]

def construir_frente(restricoes_X, u_star, x0_solver,npp,f1_star, f2_linha, f1_linha, f2_star):
   
   '''
   Loop principal
   Para cada ângulo θ_k:
      1. Calcula os pesos:  w1 = sin(θ_k),  w2 = cos(θ_k)
      2. Resolve (TR) com esses pesos
      3. Guarda o ponto encontrado
   '''
   thetas = espacamento_angulos(f1_star, f2_linha, f1_linha, f2_star, u_star, npp) 
   pontos = []
   for theta in thetas:
    w1, w2 = np.sin(theta), np.cos(theta)
    pt = tchebychev_raios_principal(w1, w2, u_star, restricoes_X, x0_solver)
    if pt is not None:
        pontos.append(pt)


   return np.array(pontos)


def roda_tchebychev_raios(restricoes, x0, npp=30, eps=0.1):

    # Passo 2 — extremos
    P1 = encontra_minimo(lambda x: x[0], restricoes, x0)
    PN = encontra_minimo(lambda x: x[1], restricoes, x0)

    # Passo 3 — ponto utópico
    u_star = np.array([P1[0] - eps, PN[1] - eps])

    # Passos 4/k — loop dos raios
    x0_z = list(x0) + [1.0]
    pontos = construir_frente(restricoes, u_star, x0_z, npp,
                              P1[0], P1[1], PN[0], PN[1])
    # Passo N — limpeza
    frente, dom = limpeza_nao_paretos(pontos)
    return frente, dom, P1, PN, u_star


npp = 30 

# Exemplo 4.4
restricoes_44 = [
    {'type': 'ineq', 'fun': lambda x: 5 * (x[1] - 1) + (x[0] - 3)**3 - 2},
    {'type': 'ineq', 'fun': lambda x: 4.5 - x[0]},
    {'type': 'ineq', 'fun': lambda x: 3.0 - x[1]},
]
x0_44 = [3.5, 2.0]
f44, d44, P1_44, PN_44, u44 = roda_tchebychev_raios(restricoes_44, x0_44, npp=npp, eps=0.1)

# Exemplo 4.5
restricoes_45 = [
    {'type': 'ineq', 'fun': lambda x: (x[0] - 0.5)**2 + (x[1] - 0.5)**2 - 0.5},
    {'type': 'ineq', 'fun': lambda x: 4.0 - x[0]**2 - x[1]**2},
    {'type': 'ineq', 'fun': lambda x: x[0]**3 + 2 * x[1] - 1.0},
]
x0_45 = [-1.0, -1.0]
f45, d45, P1_45, PN_45, u45 = roda_tchebychev_raios(restricoes_45, x0_45, npp=npp, eps=0.0)

# Exemplo 4.6 - Tem 3 variaveis não da pra fazer aqui 
restricoes_46 = []
def f46_1(x):
    return 1 - np.exp(-3 * np.sum((x - 0.577)**2))
def f46_2(x):
    return 1 - np.exp(-3 * np.sum((x + 0.577)**2))

# Exemplo 4.7
def g47(x):
    ang = np.pi / 2 if abs(x[1]) < 1e-10 else np.arctan(x[0] / x[1])
    return x[0]**2 + x[1]**2 - 1 - 0.1 * np.cos(16 * ang)

restricoes_47 = [
    {'type': 'ineq', 'fun': lambda x: g47(x)},
    {'type': 'ineq', 'fun': lambda x: 0.5 - (x[0] - 0.5)**2 - (x[1] - 0.5)**2},
]
x0_47 = [0.5, 0.5]
f47, d47, P1_47, PN_47, u47 = roda_tchebychev_raios(restricoes_47, x0_47, npp=npp, eps=0.1)


# PLOTAGEM
exemplos = [
    (f44, d44, P1_44, PN_44, u44, '#c0286a', 'Ex 4.4 — Não Convexa', 's.a. 5(x2-1)+(x1-3)^3>=2'),
    (f45, d45, P1_45, PN_45, u45, '#2a9d3f', 'Ex 4.5 — Desconectada', 's.a. restrições do exemplo 4.5'),
    (None, None, None, None, None, None, 'Ex 4.6 — Exponencial', 's.a. 3 variáveis'),
    (f47, d47, P1_47, PN_47, u47, '#6d3ebf', 'Ex 4.7 — Oscilante', 's.a. restrições do exemplo 4.7'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Tchebychev ao Longo de Raios — Capítulo 4', fontsize=14, fontweight='bold')
plt.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, ax in enumerate(axes.flat):
    frente, dom, P1, PN, u, cor, titulo, restr = exemplos[idx]
    ax.set_title(titulo, fontsize=10, fontweight='bold', loc='left')
    ax.set_xlabel('f1')
    ax.set_ylabel('f2')
    ax.grid(True, alpha=0.3)

    if frente is None:
        ax.text(0.5, 0.5, 'Exemplo com 3 variáveis\nnão plotado aqui em 2D',
                transform=ax.transAxes, ha='center', va='center')
        continue

    for i in np.linspace(0, len(frente)-1, 7, dtype=int):
        ax.plot([u[0], frente[i, 0]], [u[1], frente[i, 1]],
                '--', color='gray', lw=0.8, alpha=0.4)

    o = frente[frente[:, 0].argsort()]
    ax.plot(o[:, 0], o[:, 1], '-o', color=cor, ms=4, lw=2, label='Frente Pareto')

    if len(dom):
        ax.plot(dom[:, 0], dom[:, 1], 'x', color='red', ms=7, mew=2, label='Dominados')

    ax.plot(*P1[:2], '*', color='#d4a000', ms=12, label='P1 / PN')
    ax.plot(*PN[:2], '*', color='#d4a000', ms=12)
    ax.plot(*u[:2], 'D', color='#c0286a', ms=7, label='u*')

    ax.text(0.97, 0.97, restr, transform=ax.transAxes,
            ha='right', va='top', fontsize=8, family='monospace',
            bbox=dict(fc='#f9f9f9', ec='#ccc', boxstyle='round,pad=0.3'))

    ax.legend(fontsize=8, loc='lower right')

plt.savefig('cap4_plotagem.png', dpi=150, bbox_inches='tight')
plt.show()