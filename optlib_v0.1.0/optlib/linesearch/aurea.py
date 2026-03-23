import math
import numpy as np
from typing import Callable


def aurea(
    funcao: Callable,
    x: np.ndarray,
    d: np.ndarray,
    eps: float = 1e-5,
    rho: float = 0.1,
    bmax: float = 1e8,
    verbose: bool = False,
) -> float:
    """
    Busca de linha exata via Seção Áurea.
    Minimiza phi(t) = f(x + t*d) em relação a t >= 0.

    Baseado nos Algoritmos 4.1 e 4.2 do livro (Ribeiro & Karas, 2012).

    Parameters
    ----------
    funcao : callable  f(x) -> float
    x      : ponto atual
    d      : direção de descida
    eps    : tolerância de parada
    rho    : passo inicial para a Fase 1
    bmax   : limite máximo do intervalo
    verbose: imprime progresso se True

    Returns
    -------
    t : float — passo ótimo
    """
    theta1 = (3 - math.sqrt(5)) / 2
    theta2 = 1 - theta1

    def phi(t):
        return funcao(x + t * d)

    # Fase 1: encontrar intervalo [a, b] que contém o mínimo
    a, s, b = 0.0, rho, 2 * rho
    phi_s, phi_b = phi(s), phi(b)

    iter_f1 = 0
    while (phi_b < phi_s) and (2 * b < bmax):
        if verbose:
            print(f"[aurea F1] iter={iter_f1}: a={a:.4f}, s={s:.4f}, b={b:.4f}")
        a, s, b = s, b, 2 * b
        phi_s, phi_b = phi_b, phi(b)
        iter_f1 += 1

    # Fase 2: refinamento por seção áurea
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    phi_u, phi_v = phi(u), phi(v)

    iter_f2 = 0
    while (b - a) > eps:
        if verbose and iter_f2 % 5 == 0:
            print(f"[aurea F2] iter={iter_f2}: [{a:.6f}, {b:.6f}] (tam={b-a:.6f})")
        if phi_u < phi_v:
            b, v, u = v, u, a + theta1 * (b - a)
            phi_v, phi_u = phi_u, phi(u)
        else:
            a, u, v = u, v, a + theta2 * (b - a)
            phi_u, phi_v = phi_v, phi(v)
        iter_f2 += 1

    return (u + v) / 2
