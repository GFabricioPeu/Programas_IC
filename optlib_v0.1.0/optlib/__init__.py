from .linesearch.aurea import aurea
from .linesearch.armijo import armijo
from .unconstrained.gradient import metodo_gradiente_armijo, metodo_gradiente_exato
from .unconstrained.newton import metodo_newton_puro
from .unconstrained.conjugate import metodo_gradientes_conjugados

__version__ = "0.1.0"
__all__ = [
    "aurea", "armijo",
    "metodo_gradiente_armijo", "metodo_gradiente_exato",
    "metodo_newton_puro",
    "metodo_gradientes_conjugados",
]
