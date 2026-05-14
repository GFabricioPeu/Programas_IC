import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────────────────
# TCHEBYCHEV COM VARIÁVEL AUXILIAR t
#
# Problema (T):
#   min   max{ w1*(f1(x) - u1*),  w2*(f2(x) - u2*) }
#   s.a   x ∈ X
#
# Equivalência epigráfica — introduz variável auxiliar t:
#   min   t
#   s.a   t - w1*(x1 - u1*) >= 0
#         t - w2*(x2 - u2*) >= 0
#         x ∈ X
#
# Variável de decisão: z = [x1, x2, t]
# ─────────────────────────────────────────────────────────────────────


def tchebychev(w1, w2, u_star, restricoes_X, x0):
    """
    Resolve um passo do Tchebychev via variável auxiliar t.

    Parâmetros
    ----------
    w1, w2       : floats — pesos (w1 + w2 = 1, ambos > 0)
    u_star       : tupla (u1*, u2*) — ponto utópico
    restricoes_X : lista de dicts {'type': 'ineq'/'eq', 'fun': f}
                   escritos em x = [x1, x2]
    x0           : lista [x1_init, x2_init, t_init] — ponto inicial

    Retorna
    -------
    np.array([x1, x2]) se convergiu, None caso contrário
    """
    u1, u2 = u_star

    # Função objetivo: minimizar t = z[2]
    def objetivo(z):
        return z[2]

    # Restrições de Tchebychev
    def tcheby_1(z, w1, u1):
        return z[2] - w1 * (z[0] - u1)   # t - w1*(x1 - u1*) >= 0

    def tcheby_2(z, w2, u2):
        return z[2] - w2 * (z[1] - u2)   # t - w2*(x2 - u2*) >= 0

    rest_t = [
        {'type': 'ineq', 'fun': tcheby_1, 'args': (w1, u1)},
        {'type': 'ineq', 'fun': tcheby_2, 'args': (w2, u2)},
    ]

    # Restrições de X reescritas para z = [x1, x2, t]
    rest_X = [
        {'type': r['type'], 'fun': lambda z, f=r['fun']: f([z[0], z[1]])}
        for r in restricoes_X
    ]

    res = minimize(
        objetivo, x0,
        constraints=rest_t + rest_X,
        options={'ftol': 1e-9, 'maxiter': 2000}
    )

    return res.x[:2] if res.success else None


def gerar_frente_pareto(npp, u_star, restricoes_X, x0):
    """
    Gera npp pontos da Frente de Pareto via Tchebychev.

    Parâmetros
    ----------
    npp          : int — número de pontos desejados
    u_star       : tupla (u1*, u2*) — ponto utópico
    restricoes_X : lista de restrições em x = [x1, x2]
    x0           : ponto inicial [x1, x2, t]

    Retorna
    -------
    np.array de shape (k, 2) com os pontos encontrados (k <= npp)
    """
    pontos = []
    for i in range(1, npp + 1):   # começa em 1 para evitar w1=0
        w1 = i / npp
        w2 = 1 - w1
        pt = tchebychev(w1, w2, u_star, restricoes_X, x0)
        if pt is not None:
            pontos.append(pt)
    return np.array(pontos)


# ═════════════════════════════════════════════════════════════════════
# EXEMPLO DE USO
# Problema: min {x1, x2}  s.a  (x1-3)^2 + (x2-3)^2 <= 1
# ═════════════════════════════════════════════════════════════════════

# 1. Defina as restrições do seu problema em x = [x1, x2]
restricoes = [
    {'type': 'ineq', 'fun': lambda x: 1 - (x[0]-3)**2 - (x[1]-3)**2}
]

# 2. Define o ponto utópico (abaixo do ponto ideal)
u_star = (2.0, 2.0)

# 3. Defina o ponto inicial [x1, x2, t]
x0 = [3.0, 3.0, 1.0]

# 4. Gere a frente
PF = gerar_frente_pareto(npp=30, u_star=u_star, restricoes_X=restricoes, x0=x0)

# 5. Visualize
plt.figure(figsize=(6, 6))
plt.scatter(PF[:, 0], PF[:, 1], color='steelblue', s=60, zorder=5, label='Frente de Pareto')
theta = np.linspace(0, 2*np.pi, 300)
plt.plot(3 + np.cos(theta), 3 + np.sin(theta), 'k--', alpha=0.3, label='Regiao viavel')
plt.plot(u_star[0], u_star[1], 'r*', markersize=14, label=f'u* = {u_star}')
plt.xlabel('f1'); plt.ylabel('f2')
plt.title('Tchebychev — Frente de Pareto')
plt.legend(); plt.grid(True); plt.axis('equal')
plt.tight_layout()
plt.savefig('output/tchebychev_base.png', dpi=150)
plt.show()
print("Pronto!")