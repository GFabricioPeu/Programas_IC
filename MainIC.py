import funcaoIC as ic
import numpy as np
import matplotlib.pyplot as plt   
def main():
    '''
    Prametros do Exemplo 4.5 do livro
    x_bar = np.array([1.0, 0.0])
    d = np.array([3.0, 1.0])

    # --- Execução ---
    t_calculado = aurea_debug(f_exemplo_4_5, x_bar, d)
    t_exato = 5/11

    print(f"\n--- RESULTADO FINAL ---")
    print(f"Passo calculado pelo algoritmo: {t_calculado:.8f}")
    print(f"Passo exato (analítico):        {t_exato:.8f}")
    print(f"Erro absoluto:                  {abs(t_calculado - t_exato):.8f}")
    '''
    
    '''
    # Parâmetros do exemplo 4.10 do livro: eta=0.25, gamma=0.8 
    x0 = np.array([1.0, 0.0])
    d0 = np.array([3.0, 1.0])


    t_armijo = ic.armijo_debug(ic.f_exemplo_4_10, ic.gradiente_exemplo_4_10, x0, d0, eta=0.25, gamma=0.8)

    print(f"\nPasso final aceito: {t_armijo:.4f}")
    '''

    '''
    AMOSTRAGEM MAIS OBJETIVA
    x_inicial = np.array([1.0, 2.0]) # Ponto sugerido no Exemplo 6.1

    x_opt, hist_f, hist_g = ic.metodo_gradiente(ic.f_problema_6_1, ic.grad_problema_6_1, x_inicial)

    print(f"\nSolução encontrada: {x_opt}")
    print(f"Valor final da função: {ic.f_problema_6_1(x_opt):.8f}")

    # Gráfico de Convergência (Similar às Figuras 6.1 e 6.2 do livro)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(hist_f)
    plt.title("Convergência de f(x)")
    plt.xlabel("Iterações")
    plt.ylabel("Valor da Função")
    plt.yscale('log') # Escala logarítmica para ver melhor a descida

    plt.subplot(1, 2, 2)
    plt.plot(hist_g, color='orange')
    plt.title("Norma do Gradiente ||g||")
    plt.xlabel("Iterações")
    plt.yscale('log')

    plt.tight_layout()
    plt.show()
'''
'''
#AMOSTRAGEM EM CIMA DA FUNCAO
x_inicial = np.array([1.0, 2.0])

# Chamada da função modificada (agora pega 4 retornos)
x_opt, hist_f, hist_g, trajetoria = ic.metodo_gradiente_armijo(ic.f_problema_6_1, ic.gradiente_problema_6_1, x_inicial)

# Plotagem 
plt.figure(figsize=(10, 8))

# Cria o fundo (Curvas de Nível)
x1_min, x1_max = -0.5, 1.5
x2_min, x2_max = -0.5, 2.5
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(x2_min, x2_max, 100)
X, Y = np.meshgrid(x1_grid, x2_grid)
Z = X**2 + 4*X*Y + 6*Y**2 # Função f(x) no grid

plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.colorbar(label='Valor de f(x)')

# Desenha a trajetória
# trajetoria[:, 0] são todos os x1
# trajetoria[:, 1] são todos os x2
plt.plot(trajetoria[:, 0], trajetoria[:, 1], 'r.-', label='Caminho do Gradiente')
plt.plot(trajetoria[0, 0], trajetoria[0, 1], 'bo', label='Início')
plt.plot(trajetoria[-1, 0], trajetoria[-1, 1], 'go', label='Fim')

plt.title('Trajetória do Método do Gradiente (Zigue-Zague)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
'''
'''
x_inicial = np.array([1.0, 2.0])

# Chamada da função
trajetoria = ic.metodo_NewtonPuro(ic.Funcao_Newton_teste, ic.Gradiente_Newton_teste,ic.Hessiana_Newton_teste, x_inicial)

# Plotagem 
plt.figure(figsize=(10, 8))

# Cria o fundo (Curvas de Nível)
x1_min, x1_max = -0.5, 1.5
x2_min, x2_max = -0.5, 2.5
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(x2_min, x2_max, 100)
X, Y = np.meshgrid(x1_grid, x2_grid)
Z = X**2 + 4*X*Y + 6*Y**2 # Função f(x) no grid

plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.colorbar(label='Valor de f(x)')

# Desenha a trajetória
# trajetoria[:, 0] são todos os x1
# trajetoria[:, 1] são todos os x2
plt.plot(trajetoria[:, 0], trajetoria[:, 1], 'r.-', label='Newton')
plt.plot(trajetoria[0, 0], trajetoria[0, 1], 'bo', label='Início')
plt.plot(trajetoria[-1, 0], trajetoria[-1, 1], 'go', label='Fim')

plt.title('Trajetória do Método de Newton')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

'''
x_inicial = np.array([1.0, 2.0])

x_opt, hist_f, hist_g, trajetoria = ic.metodo_gradiente_armijo(ic.Funcao_Newton_teste,ic.Gradiente_Newton_teste,x_inicial)

# Plotagem 
plt.figure(figsize=(10, 8))

# Cria o fundo (Curvas de Nível)
x1_min, x1_max = -0.5, 1.5
x2_min, x2_max = -0.5, 2.5
x1_grid = np.linspace(x1_min, x1_max, 100)
x2_grid = np.linspace(x2_min, x2_max, 100)
X, Y = np.meshgrid(x1_grid, x2_grid)
Z = X**2 + 4*X*Y + 6*Y**2 # Função f(x) no grid

plt.contour(X, Y, Z, levels=30, cmap='viridis', alpha=0.6)
plt.colorbar(label='Valor de f(x)')

# Desenha a trajetória
# trajetoria[:, 0] são todos os x1
# trajetoria[:, 1] são todos os x2
plt.plot(trajetoria[0][:, 0], trajetoria[0][:, 1], 'r.-')
plt.plot(trajetoria[0, 0], trajetoria[0, 1], 'bo', label='Início')
plt.plot(trajetoria[-1, 0], trajetoria[-1, 1], 'go', label='Fim')

plt.title('Trajetória do Método Gradientes Conjugados')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


main()