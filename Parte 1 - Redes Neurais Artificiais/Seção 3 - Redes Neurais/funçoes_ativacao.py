import numpy as np

def stepFunction(soma):
    if (soma >= 1):
        return 1
    return 0

def sigmoidFunction(soma):
    return 1 / (1 + np.exp((-soma)))

def tanFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma)) 

def relu(soma):
    if (soma >= 0):
        return soma
    return 0

def linearFunction(soma):
    return soma
    
def softmaxFunction(x):
    ex = np.exp(x)
    return ex / ex.sum()

'''teste = stepFunction(-30)
teste = sigmoidFunction(0.358)
teste = tanFunction(0.358)
teste = relu(-0.358) 
teste = linearFunction(-0.358)
valores = [5.0, 2.0, 1.3]
print(softmaxFunction(valores))

entradas = np.array([5, 2, 1])
pesos = np.array([0.2, 0.5, 0.1])

soma = entradas.dot(pesos)
stepResultado = stepFunction(soma)
sigmoidResultado = sigmoidFunction(soma)
tanResultado = tanFunction(soma)
reluResultado = relu(soma)
linearResultado = linearFunction(soma)''' 

saidas = np.array([1, 0, 1, 0])
valoresCalculados = np.array([0.3, 0.02, 0.89, 0.32])

erroAbsoluto = np.mean(abs(saidas - valoresCalculados))
meansquaredErro = np.mean(np.square(saidas - valoresCalculados))
rootmeansquaredErro = np.sqrt(meansquaredErro)
taxadeAcerto = (1 - erroAbsoluto) * 100