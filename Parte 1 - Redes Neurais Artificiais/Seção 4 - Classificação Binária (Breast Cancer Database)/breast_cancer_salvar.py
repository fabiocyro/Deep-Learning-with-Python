import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

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

'''Como o nome do arquivo diz, aqui é onde se salva a rede neural(para não ser
necessário ficar treinando sempre) as linhas de código seguinte mostram como
fazer pra salvar os arquivos'''

classificador_json = classificador.to_json()

with open('classificador_breast.json', 'w') as json_file:
    json_file.write(classificador_json) #aqui se salva os parâmetros da RN
classificador.save_weights('classificador_breast.h5') #aqui se salva os pesos (h5 é o melhor formato
#Lembrando que caso ocorra algum erro em salvar os pesos, importar h5 (pip install h5py)

