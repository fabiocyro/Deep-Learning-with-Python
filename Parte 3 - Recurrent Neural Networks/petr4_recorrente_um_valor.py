from tensorflow.keras.layers import Dropout, Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler #importante importação para normalizar os dados e melhorar a velocidade
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("petr4_treinamento.csv")
base = base.dropna() #Very important code to remove nan values
base_treinamento = base.iloc[:, 1:2].values #usando só a primeira coluna 

normalizador = MinMaxScaler(feature_range = (0,1))

base_treinamento_normalizada = normalizador.fit_transform(base_treinamento)   

#aula seguinte

'''como a rede neural recorrente precisa de valores previos para realizar a
previsao, é necessário realizar um pre processamento nos dados antes de aplicar
na rede neural, exemplicando: pego um valor que eu quero prever, pego os 4
(usado no exemplo, na aplicação a seguir sera os 90 valores anteriores) val-
ores anteriores e o o y será o valor que eu tenho (supervised). Para maiores 
informações, checar aula 71.'''

previsores = []
preco_real = []

for i in range(90, len(base_treinamento_normalizada)):
    previsores.append(base_treinamento_normalizada[i - 90:i, 0]) #Pegandos os 90 valores anteriores ao i
    preco_real.append(base_treinamento_normalizada[i, 0])

previsores, preco_real = np.array(previsores), np.array(preco_real) #necessario converter para numpy array

previsores = np.reshape(previsores, [previsores.shape[0], previsores.shape[1], 1])

regressor = Sequential()

regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (previsores.shape[1], 1)))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True)) 
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.3))

regressor.add(LSTM(units = 50)) #return_sequences removida na ultima camada de memoria, lembrar!
regressor.add(Dropout(0.3))

regressor.add(Dense(units = 1, activation = "linear")) #sigmoid funciona também

regressor.compile(optimizer = 'rmsprop', loss = "mean_squared_error", metrics = ['mean_absolute_error'])
regressor.fit(previsores, preco_real, epochs = 100, batch_size = 32) #epochs 100 mínimo para esse tipo de rede


''' units representa a quantidade de celulas de memoria
return_sequences = True usa-se quando se tem mais de uma camada de celulas de memoria
e voce quer passar as informações adiante'''

base_teste = pd.read_csv('petr4_teste.csv')
preco_real_teste = base.iloc[:, 1:2].values

#É preciso agora pegar os 90 valores anteriores de cada valor da base teste

base_completa = pd.concat((base['Open'], base_teste['Open']), axis = 0) #Necessário fazer pois os valores do test precisam dos 90 valores anteriores. Axis = 0 significa concatenação por coluna
entradas = base_completa[len(base_completa) - len(base_teste) - 90:].values
entradas = entradas.reshape(-1, 1)
entradas = normalizador.transform(entradas)

X_teste = []
for i in range(90, 112):
    X_teste.append(entradas[i-90:i, 0])
X_teste = np.array(X_teste)
X_teste = np.reshape(X_teste, (X_teste.shape[0], X_teste.shape[1], 1))
previsoes = regressor.predict(X_teste)
previsoes = normalizador.inverse_transform(previsoes) #para "desnormalizar" e visualizar os valores reais

previsoes.mean()
preco_real_teste.mean()

plt.plot(preco_real_teste, color = 'red', label = 'PreÃ§o real')
plt.plot(previsoes, color = 'blue', label = 'Previsões')
plt.title('Previsão preço das ações')
plt.xlabel('Tempo')
plt.ylabel('Valor Yahoo')
plt.legend()
plt.show()