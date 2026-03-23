import numpy as np
from typing import Callable


def armijo(
    funcao: Callable,
    gradiente_funcao: Callable,
    x: np.ndarray,
    d: np.ndarray,
    gamma: float = 0.7,
    eta: float = 0.45,
    verbose: bool = False,
) -> float:
    """
    Busca de linha inexata — condição de Armijo (decaimento suficiente).

    Baseado no Algoritmo 4.3 do livro (Ribeiro & Karas, 2012).

    Parameters
    ----------
    funcao           : callable  f(x) -> float
    gradiente_funcao : callable  grad_f(x) -> np.ndarray
    x                : ponto atual
    d                : direção de descida
    gamma            : fator de redução do passo  (0 < gamma < 1)
    eta              : exigência de decaimento    (0 < eta   < 1)
    verbose          : imprime progresso se True

    Returns
    -------
    t : float — passo aceito
    """
    t = 1.0
    f_atual = funcao(x)
    gd = np.dot(gradiente_funcao(x), d)

    if verbose:
        print(f"[armijo] f(x)={f_atual:.4f} | g^T d={gd:.4f}")

    k = 0
    while funcao(x + t * d) > f_atual + eta * t * gd:
        if verbose:
            print(f"[armijo] iter={k}: t={t:.4f} RECUSADO")
        t *= gamma
        k += 1

    if verbose:
        print(f"[armijo] iter={k}: t={t:.4f} ACEITO")
    return t
