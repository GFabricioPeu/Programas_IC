# 📐 Programas_IC — Biblioteca de Otimização Não Linear

Repositório de estudo e implementação de algoritmos de otimização não linear, desenvolvido durante Iniciação Científica. Os algoritmos são baseados no livro **"Otimização Contínua"** (Ribeiro & Karas, 2012).

---

## 📁 Estrutura do Projeto

```
Programas_IC/
├── optlib_v0.1.0/
│   ├── optlib/
│   │   ├── linesearch/
│   │   │   ├── aurea.py        # Busca exata — Seção Áurea (Alg. 4.1/4.2)
│   │   │   └── armijo.py       # Busca inexata — Backtracking Armijo (Alg. 4.3)
│   │   ├── unconstrained/
│   │   │   ├── gradient.py     # Método do Gradiente c/ Armijo e c/ Áurea (Alg. 6.1)
│   │   │   ├── newton.py       # Método de Newton Puro (Alg. 7.1)
│   │   │   └── conjugate.py    # Gradientes Conjugados — Fletcher-Reeves (Alg. 8.1)
│   │   └── utils/
│   │       └── problems.py     # Funções de teste dos exemplos do livro
│   └── setup.py
├── MainIC.py                   # Script principal com exemplos de uso
├── .gitignore
└── README.md
```

---

## ⚙️ Instalação

```bash
cd optlib_v0.1.0
pip install -e .
```

A flag `-e` instala em modo editável — qualquer modificação nos arquivos reflete imediatamente, sem reinstalar.

**Verificar instalação:**
```bash
python -c "import optlib; print(optlib.__version__)"
# 0.1.0
```

---

## 🚀 Uso Rápido

```python
import optlib as ic
import numpy as np

x0 = np.array([1.0, 2.0])

# Método do Gradiente com Armijo
x_opt, hist_f, hist_g, traj = ic.metodo_gradiente_armijo(
    ic.f_problema_6_1, ic.grad_problema_6_1, x0)

# Método de Newton Puro
x_opt, hist_f, hist_g, traj = ic.metodo_newton_puro(
    ic.f_quadratica, ic.grad_quadratica, ic.hess_quadratica, x0)

# Gradientes Conjugados (Fletcher-Reeves)
x_opt, hist_f, hist_g, traj = ic.metodo_gradientes_conjugados(
    ic.f_quadratica, ic.grad_quadratica, x0)
```

Todos os métodos retornam a mesma estrutura:
```python
x_opt      # solução encontrada
hist_f     # lista de f(x) por iteração
hist_g     # lista de ||grad f(x)|| por iteração
trajetoria # array (n_iters × n) com todos os pontos visitados
```

---

## 📚 Algoritmos Implementados

### Busca em Linha (`optlib.linesearch`)

| Função | Descrição | Referência |
|--------|-----------|------------|
| `aurea(f, x, d)` | Minimiza φ(t) = f(x + td) via Seção Áurea | Alg. 4.1/4.2 |
| `armijo(f, grad_f, x, d)` | Backtracking com condição de Armijo | Alg. 4.3 |

Ambas aceitam `verbose=True` para imprimir o progresso iteração a iteração.

### Otimização Irrestrita (`optlib.unconstrained`)

| Função | Descrição | Referência |
|--------|-----------|------------|
| `metodo_gradiente_armijo(f, grad, x0)` | Gradiente com busca inexata | Alg. 6.1 |
| `metodo_gradiente_exato(f, grad, x0)` | Gradiente com busca exata | Alg. 6.1 |
| `metodo_newton_puro(f, grad, hess, x0)` | Newton com passo unitário | Alg. 7.1 |
| `metodo_gradientes_conjugados(f, grad, x0)` | Fletcher-Reeves com Áurea | Alg. 8.1 |

### Funções de Teste (`optlib.utils.problems`)

| Função | Expressão |
|--------|-----------|
| `f_quadratica(x)` | x₁² + 4x₁x₂ + 6x₂² |
| `grad_quadratica(x)` | Gradiente analítico |
| `hess_quadratica(x)` | Hessiana analítica |
| `f_exemplo_4_5(x)` | 0.5(x₁−2)² + (x₂−1)² |
| `f_exemplo_4_10(x)` | 0.5(x₁−2)² + (x₂−1)² |
| `grad_exemplo_4_10(x)` | Gradiente analítico |
| `f_problema_6_1(x)` | Alias de `f_quadratica` |
| `grad_problema_6_1(x)` | Alias de `grad_quadratica` |

---

## 📖 Referências

- RIBEIRO, A. A.; KARAS, E. W. **Otimização Contínua**. 2012.
- ISOTON, C. **Condições de Otimalidade para Problemas Escalar e Vetorial**. UFPR, 2013.
- GUILLEN, F. **Tchebychev Scalarization Along Rays for Bi-Objective Pareto Fronts**. 2024.

---

## 🗺️ Roadmap

- [x] Busca em linha (Áurea, Armijo)
- [x] Método do Gradiente
- [x] Método de Newton Puro
- [x] Gradientes Conjugados (Fletcher-Reeves)
- [ ] Quasi-Newton (BFGS)
- [ ] Região de Confiança
- [ ] Métodos com Restrições (Penalidade, SQP)
- [ ] Otimização Multiobjetivo (Tchebychev, Pareto)
