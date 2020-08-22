'''Esse exercício consistiu em procurar obter uma precisão de 90% ou mais
(a variável media). Para este caso, foi obtido por adicionar uma camada escondida
extra, com 16 neurons, adam como otimizador e 400 epochs.
obs: Detalhe para o desvio padrão baixo o que significa uma baixa possibilidade
de overfitting'''

import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')



def criarRede():
    
    classificador = Sequential() 
    classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform", input_dim = 30)) 
    classificador.add(Dropout(0.2)) #Zera alguns valores da entrada, de modo a eviter overfitting (valores muito altos podem causar underfitting)
    classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform")) 
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform")) 
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units = 1, activation = "sigmoid"))


    #otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.6) 


    classificador.compile(optimizer = 'adam', loss = "binary_crossentropy",
                     metrics = ["binary_accuracy"])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                 epochs = 400,
                                 batch_size = 10)

resultados = cross_val_score(classificador,
                             X = previsores, y = classes,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std() #Quanto MAIOR, mais a possibilidade de overfitting

