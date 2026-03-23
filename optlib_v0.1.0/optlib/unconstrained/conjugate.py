import numpy as np
from typing import Callable, Tuple
from ..linesearch.aurea import aurea


def metodo_gradientes_conjugados(
    funcao: Callable,
    gradiente: Callable,
    x0: np.ndarray,
    stop: float = 1e-5,
    max_iter: int = 1000,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, list, np.ndarray]:
    """
    Método dos Gradientes Conjugados — fórmula Fletcher-Reeves.

    Busca de linha exata via Seção Áurea.
    Baseado no Algoritmo 8.1 (Ribeiro & Karas, 2012).

    Returns
    -------
    x_opt, historico_f, historico_g, trajetoria
    """
    x = x0.copy()
    historico_f, historico_g = [], []
    trajetoria = [x.copy()]

    g = gradiente(x)
    d = -g
    beta = 0.0

    if verbose:
        print(f"{'Iter':<5} | {'f(x)':<12} | {'||g||':<12} | {'beta':<10}")
        print("-" * 48)

    for k in range(max_iter):
        norm_g = np.linalg.norm(g)
        f_val = funcao(x)
        historico_f.append(f_val)
        historico_g.append(norm_g)

        if verbose:
            print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f} | {beta:<10.6f}")

        if norm_g < stop:
            if verbose:
                print(f"\nCONVERGIU em {k} iterações.")
            return x, historico_f, historico_g, np.array(trajetoria)

        t = aurea(funcao, x, d)
        x_novo = x + t * d
        g_novo = gradiente(x_novo)

        # Fletcher-Reeves: beta = ||g_novo||^2 / ||g||^2
        beta = np.dot(g_novo, g_novo) / np.dot(g, g)
        d = -g_novo + beta * d

        x, g = x_novo, g_novo
        trajetoria.append(x.copy())

    if verbose:
        print("Atingiu número máximo de iterações.")
    return x, historico_f, historico_g, np.array(trajetoria)
