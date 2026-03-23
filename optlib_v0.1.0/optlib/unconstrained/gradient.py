import numpy as np
from typing import Callable, Tuple
from ..linesearch.armijo import armijo
from ..linesearch.aurea import aurea


def metodo_gradiente_armijo(
    funcao: Callable,
    gradiente_funcao: Callable,
    x0: np.ndarray,
    stop: float = 1e-5,
    max_iter: int = 1000,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, list, np.ndarray]:
    """
    Método do Gradiente com busca de Armijo.

    Baseado no Algoritmo 6.1 (Ribeiro & Karas, 2012).

    Returns
    -------
    x_opt      : solução encontrada
    historico_f: lista de f(x) por iteração
    historico_g: lista de ||grad f(x)|| por iteração
    trajetoria : array (n_iters, n) com todos os pontos visitados
    """
    x = x0.copy()
    historico_f, historico_g = [], []
    trajetoria = [x.copy()]

    if verbose:
        print(f"{'Iter':<5} | {'f(x)':<12} | {'||g||':<12} | {'t':<10}")
        print("-" * 48)

    for k in range(max_iter):
        g = gradiente_funcao(x)
        norm_g = np.linalg.norm(g)
        f_val = funcao(x)
        historico_f.append(f_val)
        historico_g.append(norm_g)

        if norm_g < stop:
            if verbose:
                print(f"\nCONVERGIU em {k} iterações.")
            return x, historico_f, historico_g, np.array(trajetoria)

        d = -g
        t = armijo(funcao, gradiente_funcao, x, d)

        if verbose:
            print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f} | {t:<10.4f}")

        x = x + t * d
        trajetoria.append(x.copy())

    if verbose:
        print("Atingiu número máximo de iterações.")
    return x, historico_f, historico_g, np.array(trajetoria)


def metodo_gradiente_exato(
    funcao: Callable,
    gradiente_funcao: Callable,
    x0: np.ndarray,
    stop: float = 1e-5,
    max_iter: int = 1000,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, list, np.ndarray]:
    """
    Método do Gradiente com busca exata (Seção Áurea).

    Returns
    -------
    x_opt, historico_f, historico_g, trajetoria
    """
    x = x0.copy()
    historico_f, historico_g = [], []
    trajetoria = [x.copy()]

    if verbose:
        print(f"{'Iter':<5} | {'f(x)':<12} | {'||g||':<12} | {'t':<10}")
        print("-" * 48)

    for k in range(max_iter):
        g = gradiente_funcao(x)
        norm_g = np.linalg.norm(g)
        f_val = funcao(x)
        historico_f.append(f_val)
        historico_g.append(norm_g)

        if norm_g < stop:
            if verbose:
                print(f"\nCONVERGIU em {k} iterações.")
            return x, historico_f, historico_g, np.array(trajetoria)

        d = -g
        t = aurea(funcao, x, d)

        if verbose:
            print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f} | {t:<10.4f}")

        x = x + t * d
        trajetoria.append(x.copy())

    if verbose:
        print("Atingiu número máximo de iterações.")
    return x, historico_f, historico_g, np.array(trajetoria)
