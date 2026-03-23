import numpy as np
from typing import Callable, Tuple


def metodo_newton_puro(
    funcao: Callable,
    gradiente: Callable,
    hessiana: Callable,
    x0: np.ndarray,
    stop: float = 1e-5,
    max_iter: int = 100,
    verbose: bool = True,
) -> Tuple[np.ndarray, list, list, np.ndarray]:
    """
    Método de Newton Puro (direção d = -H^{-1} g, passo t=1).

    Baseado no Algoritmo 7.1 (Ribeiro & Karas, 2012).

    Returns
    -------
    x_opt, historico_f, historico_g, trajetoria
    """
    x = x0.copy()
    historico_f, historico_g = [], []
    trajetoria = [x.copy()]

    if verbose:
        print(f"{'Iter':<5} | {'f(x)':<12} | {'||g||':<12}")
        print("-" * 35)

    for k in range(max_iter):
        g = gradiente(x)
        f_val = funcao(x)
        norm_g = np.linalg.norm(g)
        historico_f.append(f_val)
        historico_g.append(norm_g)

        if verbose:
            print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f}")

        if norm_g < stop:
            if verbose:
                print(f"\nCONVERGIU em {k} iterações.")
            return x, historico_f, historico_g, np.array(trajetoria)

        H = hessiana(x)
        d = np.linalg.solve(H, -g)
        x = x + d
        trajetoria.append(x.copy())

    if verbose:
        print("Atingiu número máximo de iterações.")
    return x, historico_f, historico_g, np.array(trajetoria)
