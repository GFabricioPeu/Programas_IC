"""
Funções de teste para os algoritmos de otimização.
Baseadas nos exemplos do livro (Ribeiro & Karas, 2012).
"""
import numpy as np


# f(x) = x1^2 + 4*x1*x2 + 6*x2^2  (Exemplo 6.1 / Newton teste)
def f_quadratica(x):
    return x[0]**2 + 4*x[0]*x[1] + 6*x[1]**2

def grad_quadratica(x):
    return np.array([2*x[0] + 4*x[1], 4*x[0] + 12*x[1]])

def hess_quadratica(x):
    return np.array([[2.0, 4.0], [4.0, 12.0]])


# Exemplo 4.5 / 4.10
def f_exemplo_4_5(x):
    return 0.5 * (x[0] - 2)**2 + (x[1] - 1)**2

def f_exemplo_4_10(x):
    return 0.5 * (x[0] - 2)**2 + (x[1] - 1)**2

def grad_exemplo_4_10(x):
    return np.array([x[0] - 2, 2*(x[1] - 1)])


# Problema 6.1 (alias mantido para compatibilidade)
def f_problema_6_1(x):
    return f_quadratica(x)

def grad_problema_6_1(x):
    return grad_quadratica(x)
