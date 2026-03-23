import numpy as np
import math
import matplotlib.pyplot as plt

#f(x) Newton teste = x1^2 + 4x1*x2 + 6x2^2
def Funcao_Newton_teste(x):
    return x[0]**2 + 4*x[0]*x[1] + 6*x[1]**2

def Gradiente_Newton_teste(x):
    return np.array([2*x[0]+4*x[1],4*x[0]+12*x[1]])

def Hessiana_Newton_teste(x):
    return np.array ([[2.0,4.0],[4.0,12.0]])

"""def avalia_f_derivada(x,ordem):
    x = np.array(x)
    if ordem == 0:
    
        return x**2 + 2*x 
"""

def f_exemplo_4_10(x):
    return 0.5 * (x[0] - 2)**2 + (x[1] - 1)**2

def gradiente_exemplo_4_10(x):
    # Derivada de 0.5*(x-2)^2 é (x-2)
    # Derivada de (y-1)^2 é 2*(y-1)
    return np.array([ (x[0] - 2), 2 * (x[1] - 1) ])

def f_exemplo_4_5(x):
    # f(x) = 0.5 * (x1 - 2)^2 + (x2 - 1)^2
    return 0.5 * (x[0] - 2)**2 + (x[1] - 1)**2

def f_problema_6_1(x):
    return x[0]**2 + 4*x[0]*x[1] + 6*x[1]**2

def gradiente_problema_6_1(x):
    df_dx1 = 2*x[0] + 4*x[1]
    df_dx2 = 4*x[0] + 12*x[1]
    return np.array([df_dx1, df_dx2])

def aurea(funcao, x, d, eps=1e-5, rho=0.1, bmax=1e8):
    """
    Encontra o t que minimiza f(x + t*d) usando a razão áurea.
    """
    # Constantes definidas no livro
    theta1 = (3 - math.sqrt(5)) / 2
    theta2 = 1 - theta1
    
    # funcaoção auxiliar phi(t) = f(x + t*d)
    def phi(t):
        return funcao(x + t * d)

    # --- Fase 1: Obtenção do intervalo [a, b] ---
    a = 0
    s = rho
    b = 2 * rho
    
    phi_b = phi(b)
    phi_s = phi(s)
    
    # Expande o intervalo enquanto a funcaoção estiver decrescendo
    while (phi_b < phi_s) and (2*b < bmax):
        a = s
        s = b
        b = 2 * b
        phi_s = phi_b
        phi_b = phi(b)
    
    # --- Fase 2: Redução do intervalo (Refinamento) ---
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    
    phi_u = phi(u)
    phi_v = phi(v)
    
    while (b - a) > eps:
        if phi_u < phi_v:
            b = v
            v = u
            u = a + theta1 * (b - a)
            phi_v = phi_u
            phi_u = phi(u)
        else:
            a = u
            u = v
            v = a + theta2 * (b - a)
            phi_u = phi_v
            phi_v = phi(v)
            
    # Retorna o ponto médio do intervalo final
    t = (u + v) / 2
    return t

def aurea_debug(funcao, x, d, eps=1e-5, rho=0.1, bmax=1e8):
    theta1 = (3 - math.sqrt(5)) / 2
    theta2 = 1 - theta1
    
    def phi(t):
        return funcao(x + t * d)

    print(f"--- FASE 1: Encontrando o intervalo [a, b] ---")
    a = 0
    s = rho
    b = 2 * rho
    
    phi_a = phi(a)
    phi_s = phi(s)
    phi_b = phi(b)
    
    # Expansão para cercar o mínimo
    iter_f1 = 0
    while (phi_b < phi_s) and (2*b < bmax):
        print(f"Iter {iter_f1}: a={a:.4f}, s={s:.4f}, b={b:.4f} | f(b)={phi_b:.4f} < f(s)={phi_s:.4f}")
        a = s
        s = b
        b = 2 * b
        phi_s = phi_b
        phi_b = phi(b)
        iter_f1 += 1
    
    print(f"Intervalo inicial encontrado: [{a:.4f}, {b:.4f}]")

    # --- FASE 2: Redução do intervalo ---
    print(f"\n--- FASE 2: Refinando com Seção Áurea ---")
    u = a + theta1 * (b - a)
    v = a + theta2 * (b - a)
    
    phi_u = phi(u)
    phi_v = phi(v)
    
    iter_f2 = 0
    while (b - a) > eps:
        # Mostra o progresso a cada 5 iterações para não poluir
        if iter_f2 % 5 == 0: 
             print(f"Iter {iter_f2}: Intervalo [{a:.6f}, {b:.6f}] (tamanho: {b-a:.6f})")
             
        if phi_u < phi_v:
            b = v
            v = u
            u = a + theta1 * (b - a)
            phi_v = phi_u
            phi_u = phi(u) # Reutiliza v, calcula novo u
        else:
            a = u
            u = v
            v = a + theta2 * (b - a)
            phi_u = phi_v
            phi_v = phi(v) # Reutiliza u, calcula novo v
        iter_f2 += 1
            
    t = (u + v) / 2
    return t

def armijo(funcao, gradiente_funcao, x, d, gamma=0.7, eta=0.45):
    """
    Encontra um t que satisfaz a condição de decaimento suficiente.
    """
    t = 1.0
    
    f_atual = funcao(x)
    g_atual = gradiente_funcao(x)
    
    # Produto interno do gradiente pela direção (g^T * d)
    gd = np.dot(g_atual, d)
    
    # Loop de backtracking: reduz t até satisfazer a condição
    # f(x + td) <= f(x) + eta * t * (gradiente^T * d)
    while funcao(x + t * d) > f_atual + eta * t * gd:
        t = gamma * t
        
    return t

def armijo_debug(funcao, gradiente_funcao, x, d, eta=0.25, gamma=0.8):
    """
    funcao: A funcaoção f(x)
    gradiente_funcao: O gradiente de f(x)
    x: Ponto atual
    d: Direção de descida
    eta: Parâmetro de exigência (0 < eta < 1)
    gamma: Fator de redução do passo (0 < gamma < 1)
    """
    t = 1.0 # Começamos tentando dar um passo completo (Newton gosta disso)
    
    f_atual = funcao(x)
    g_atual = gradiente_funcao(x)
    
    # Produto interno do gradiente pela direção (g^T * d)
    # Isso diz o quão inclinada é a descida nessa direção
    gd = np.dot(g_atual, d) 
    
    print(f"--- INÍCIO ARMIJO ---")
    print(f"f(x) inicial: {f_atual:.4f}")
    print(f"Inclinação direcional (g^T d): {gd:.4f}")
    
    iteracao = 0
    # Loop de Backtracking
    while True:
        # 1. Calculamos o valor no ponto candidato
        x_novo = x + t * d
        f_novo = funcao(x_novo)
        
        # 2. Calculamos o limite aceitável (A reta da condição de Armijo)
        limite_aceitavel = f_atual + eta * t * gd
        
        # 3. Testamos
        aceito = f_novo <= limite_aceitavel
        status = "ACEITO" if aceito else "RECUSADO"
        
        print(f"Iter {iteracao}: t={t:.4f} | f(novo)={f_novo:.4f} | Teto={limite_aceitavel:.4f} -> {status}")
        
        if aceito:
            break # Sai do loop se a condição for satisfeita
            
        # Se recusado, reduz o passo
        t = gamma * t
        iteracao += 1
        
    return t

def metodo_NewtonPuro(funcao,gradiente,hessiana,x0,stop=1e-5,max_iter = 100):
    x= x0.copy()
    trajetoria = [x.copy()]

    print(f"\n---Newton---")
    print(f"{'Iter':<5}|{'f(x)':<12}|{'||grad||':<12}")

    for k in range(max_iter):
        g = gradiente(x)
        f_valor = funcao(x)
        norma_gradiente = np.linalg.norm(g)

        print(f"{k:<5} | {f_valor:<12.6f} | {norma_gradiente:<12.6f}")
        
        if norma_gradiente < stop:
            print(f"\nConvergiu em {k} iteracoes")
            return np.array(trajetoria)
        
        H = hessiana(x)
        d = np.linalg.solve(H,-g)

        x = x + d
        trajetoria.append(x.copy())

    return np.array(trajetoria)

def metodo_gradiente_armijo(funcao, gradiente_funcao, x0, stop=1e-5, max_iter=100):
    """
    Minimiza f(x) usando direção d = -gradiente e busca de Armijo.
    Baseado no Algoritmo 5.1 e Rotina 6.6.
    """
    x = x0.copy()
    historico_f = []  # Para guardar os valores e plotar depois
    historico_g = []  # Para guardar a norma do gradiente
    trajetoria = [x.copy()] # Guarda os pontos x visitados
    
    print(f"{'Iter':<5} | {'f(x)':<12} | {'||gradiente||':<12} | {'Passo (t)':<10}")
    print("-" * 50)
    
    for k in range(max_iter):
        g = gradiente_funcao(x)
        norm_g = np.linalg.norm(g)
        f_val = funcao(x)
        
        historico_f.append(f_val)
        historico_g.append(norm_g)
        
        # 1. Critério de Parada (Seção 4.3.2 - Ponto Estacionário)
        if norm_g < stop:
            print("-" * 50)
            print(f"CONVERGIU em {k} iterações!")
            return x, historico_f, historico_g,np.array(trajetoria)
        
        # 2. Definição da Direção (Seção 4.1 - Descida)
        # No método do gradiente, a direção é oposta ao gradiente
        d = -g 
        
        # 3. Cálculo do Passo (Seção 4.2.2 - Armijo)
        t = armijo(funcao, gradiente_funcao, x, d)
        
        # Print de progresso
        print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f} | {t:<10.4f}")
        
        # 4. Atualização
        x = x + t * d
        
        #Salva o novo ponto na lista
        trajetoria.append(x.copy())

    print("Atingiu número máximo de iterações.")
    return x, historico_f, historico_g, np.array(trajetoria)

def metodo_gradiente_exato(funcao, gradiente_funcao, x0, stop=1e-5, max_iter=1000):
    x = x0.copy()
    
    # Listas para guardar o histórico 
    historico_f = []
    historico_g = []
    trajetoria = [x.copy()]
    
    print(f"{'Iter':<5} | {'f(x)':<12} | {'||gradiente||':<12} | {'Passo (t)':<10}")
    print("-" * 50)
    
    for k in range(max_iter):
        g = gradiente_funcao(x)
        norm_g = np.linalg.norm(g)
        f_val = funcao(x)
        
        # Guarda os históricos
        historico_f.append(f_val)
        historico_g.append(norm_g)
        
        if norm_g < stop:
            print("-" * 50)
            print(f"CONVERGIU em {k} iterações!")
            
            return x, historico_f, historico_g, np.array(trajetoria)
            
        d = -g
        # Usa a busca exata (Seção Áurea)
        t = aurea(funcao, x, d)
        
        print(f"{k:<5} | {f_val:<12.6f} | {norm_g:<12.6f} | {t:<10.4f}")
        
        x = x + t * d
        trajetoria.append(x.copy())
        
    print("Atingiu número máximo de iterações.")
   
    return x, historico_f, historico_g, np.array(trajetoria)

def metodo_Gradientes_conjugados(funcao, gradiente, x0, stop=1e-5, max_iter=1000, verbose=False):
    x= x0.copy()
    trajetoria = [x.copy()]

    g = gradiente(x)
    d = -g
    print(f"\n---Gradientes Conjugados---")
    print(f"{'Iter':<5}|{'f(x)':<12}|{'||grad||':<12}")

    for k in range(max_iter):
        norma_g = np.linalg.norm(g)
        f_valor = funcao(x)

        if k == 0: beta = 0

        print(f"{k:<5} | {f_valor:<12.6f} | {norma_g:<12.6f} | {beta}")
        if norma_g < stop:
            print(f"Convergiu em {k} iteracoes!")
            return np.array(trajetoria)
        
        t = aurea(funcao,x,d)

        x_novo = x + t * d
        g_novo = gradiente(x_novo)

        beta = np.dot(g_novo,g_novo)/ np.dot(g,g)

        d_novo = -g_novo + beta * d

        x = x_novo
        g = g_novo
        d = d_novo
        trajetoria.append(x.copy())

    return x, historico_f, historico_g, np.array(trajetoria)

