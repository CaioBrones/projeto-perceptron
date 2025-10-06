import numpy as np

# Função de ativação
def step_function(x):
    return 1 if x >= 0 else -1

# Dados do problema 
# Formato: [precipitacao, pressao, radiacao, temp_max, temp_min]
X = np.array([
    [0, 945.8, 1227.8, 30.3, 29.4],
    [0, 946.6, 3142.9, 30.4, 29.2],
    [0, 946.9, 2775.2, 31.3, 30.3],
    [0, 946.9, 1607.1, 31.5, 26.6],
    [0, 947.0, 208.5, 20.4, 19.7],
    [2.2, 947.1, 179.9, 29.5, 20.4],
    [0, 947.1, 3990.6, 30.2, 28.5],
    [0, 947.3, 67.3, 24.0, 22.5],
    [0, 947.3, 0.0, 18.3, 17.8],
    [0, 947.4, 303.8, 26.6, 24.8],
    [0, 947.5, 0.0, 18.3, 17.6],
    [5.0, 947.5, 51.3, 20.7, 19.6],
    [0, 947.5, 98.8, 20.7, 19.7],
    [0, 947.5, 459.2, 24.1, 23.2],
    [0.2, 947.6, 415.3, 23.5, 21.8],
    [0.6, 947.9, 0.0, 21.2, 20.2],
    [0, 947.9, 0.0, 18.8, 18.3],
    [0, 948.0, 345.5, 25.7, 25.0],
    [0, 948.0, 0.0, 19.9, 19.6],
    [0, 948.0, 1255.3, 26.7, 24.4],
    [0, 948.1, 0.0, 18.9, 18.5],
    [0, 948.3, 3435.3, 31.6, 29.8],
    [0, 948.3, 0.0, 19.7, 18.8],
    [0, 948.4, 36.9, 25.0, 23.6]
])

# Saídas esperadas - CLASSIFICAÇÃO: 1 se umidade >= 70, -1 caso contrário
y = np.array([-1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, 1, 1, -1, 1, -1, 1, -1, 1, -1])

# Hiperparâmetros
lr = 0.4
n_epochs = 10

# Pesos iniciais (5 pesos)
pesos = np.array([0.4, -0.6, 0.6, -0.2, 0.3], dtype=float)
bias = 0.5

print("Pesos iniciais:", pesos, "Bias inicial:", bias)

# Treinamento
for epoca in range(n_epochs):
    print(f"\nÉpoca {epoca+1}")
    for x_i, y_i in zip(X, y):
        soma = np.dot(pesos, x_i) + bias
        y_pred = step_function(soma)
        erro = y_i - y_pred
        
        # Atualização dos pesos e bias
        pesos += lr * erro * x_i
        bias += lr * erro
        
        print(f"Entrada: {x_i}, Esperado: {y_i}, Previsto: {y_pred}, Erro: {erro}")
        print("Novos pesos:", pesos, "Novo bias:", bias)

print("\nPesos finais:", pesos, "Bias final:", bias)

# Predição em novos exemplo
X_teste = np.array([
    [1.5, 947.0, 1500.0, 27.0, 25.0],
    [0.0, 946.5, 2000.0, 28.0, 26.0],  
    [3.0, 947.2, 500.0, 22.0, 21.0],
    [0.1, 948.0, 3000.0, 30.0, 29.0],
    [0.2, 953.8, 2500.0, 20.0, 21.0],
    [3.5, 956.2, 1500.0, 31.0, 29.0],
    [0.3, 941.1, 4000.0, 24.0, 34.0],
    [1.0, 949.0, 500.0, 25.0, 18.0],
    [0.0, 948.6, 3500.0, 22.0, 23.0]
])

# Saídas esperadas
Y_teste = np.array([1, -1, 1, -1, -1, -1, -1, -1, -1])

print("\n--- Predições ---")
acertou = 0
exemplo_teste = 0
for x_i, y_i in zip(X_teste, Y_teste):
    soma = np.dot(pesos, x_i) + bias
    y_pred = step_function(soma)
    print(f"Entrada: {x_i} -> Saída prevista: {y_pred}")
    erro = y_i - y_pred
    if erro == 0:
        acertou += 1
    exemplo_teste += 1

acuracia = acertou / exemplo_teste
print(f"Acurácia: {acuracia}")