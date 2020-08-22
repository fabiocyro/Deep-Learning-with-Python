import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

'''Parâmetros escolhidos de acordo com o teste realizado no arquivo anterior,
(breast_cancer_tunning)'''

classificador = Sequential() 
classificador.add(Dense(units = 8, activation = 'relu',
                    kernel_initializer = 'normal', input_dim = 30)) 
classificador.add(Dropout(0.2)) #Zera alguns valores da entrada, de modo a eviter overfitting (valores muito altos podem causar underfitting)
classificador.add(Dense(units = 8, activation = 'relu',
                    kernel_initializer = 'normal')) 
classificador.add(Dropout(0.2))

classificador.add(Dense(units = 1, activation = "sigmoid"))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                 metrics = ["binary_accuracy"])

classificador.fit(previsores, classes, batch_size = 10, epochs = 100)

'''Suponde que já se sabe os melhores parâmetros para a RN, esse é o código necessário
para se realizar a implentação, o código seguinte é aonde se preenche com o que se quer saber'''

novo = np.array([[15.85, 8.34, 118, 900, 0.1, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185, 0.84,
                  158, 0.363]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)
 