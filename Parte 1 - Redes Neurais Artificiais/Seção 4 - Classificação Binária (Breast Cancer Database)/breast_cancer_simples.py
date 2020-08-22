import pandas as pd

previsores = pd.read_csv("entradas_breast.csv")
classe = pd.read_csv("saidas_breast.csv")

from sklearn.model_selection import train_test_split
previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size = 0.25)

import keras
from keras.models import Sequential #Forward propagation!
from keras.layers import Dense #Fully Connected!

classificador = Sequential() 
classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform", input_dim = 30)) 
''' comentarios sobre essa primeira configuração da Dense (camada oculta):
    units = quantidade de neuronios na camada oculta, segue a equação (entradas + saidas / 2)
    activation = tipo de ativação
    kernel_initializer = determinar a inicialização dos pesos 
    input_dim = a quantidade de atributos (utilizado apenas na camada de entrada'''

classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform")) #adicionando uma nova camada escondida 


classificador.add(Dense(units = 1, activation = "sigmoid"))

'''comentarios sobre a configuração da camada de saida:
    units = 1 pois o resultado só pode ser ou 0 ou 1, ou seja apenas um neurônio para a camada de saída
    sigmoid foi utilizado justamente por isso, importante saber 
    qual função escolher pra camada de saída'''

otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) 

''' Comentarios sobre a configuração do otimizador: 
    decay = quantidade que o lr vai diminuindo (interessante pois pode acelerar o processo)
    clipvalue = segura o valor dos pesos pra nao ficarem muito dispersos'''

classificador.compile(optimizer = otimizador, loss = "binary_crossentropy",
                     metrics = ["binary_accuracy"])

#classificador.compile(optimizer = "adam", loss = "binary_crossentropy",
 #                     metrics = ["binary_accuracy"])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 10, epochs = 100) #Treinamento

pesos0 = classificador.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = classificador.layers[1].get_weights()
pesos2 = classificador.layers[2].get_weights()


previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score  # usando o Scikit Learning para avaliar
precisao = accuracy_score(classe_teste, previsoes)
matriz = confusion_matrix(classe_teste, previsoes)

resultado = classificador.evaluate(previsores_teste, classe_teste) # Usando Keras para avaliar
