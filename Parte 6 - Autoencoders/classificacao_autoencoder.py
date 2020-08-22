import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense
from keras.utils import np_utils

(previsores_treinamento, classe_treinamento), (previsores_teste, classe_teste) = mnist.load_data()

classe_dummy_treinamento = np_utils.to_categorical(classe_treinamento)
classe_dummy_teste = np_utils.to_categorical(classe_teste)


previsores_treinamento = previsores_treinamento.astype("float32") / 255 #normalização
previsores_teste = previsores_teste.astype("float32") / 255


previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento)), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape((len(previsores_teste)), np.prod(previsores_teste.shape[1:]))



autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = previsores_treinamento.shape[1]))
autoencoder.add(Dense(previsores_treinamento.shape[1], activation = "sigmoid"))
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, 
                epochs = 50, batch_size = 256, #Number of epochs é bem importante para AE, quanto mais melhor
                validation_data=(previsores_teste, previsores_teste))


dimensao_original = Input(shape = previsores_treinamento.shape[1],)
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))

previsores_treinamento_codificados = encoder.predict(previsores_treinamento)
previsores_teste_codificados = encoder.predict(previsores_teste)

'''a ideia a seguir é criar duas redes neurais densas e comparar o desempenho
com os previsores treinamento normais (60k, 784) e os previsores_treinamento_
codificados (60k, 32) e avaliar os resultados.'''

#Sem redução de dimensionalidade
c1 = Sequential()
c1.add(Dense(units = 397, activation = 'relu', input_dim = previsores_treinamento.shape[1]))
c1.add(Dense(units = 397, activation = 'relu'))
c1.add(Dense(units = 10, activation = 'softmax'))
c1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
c1.fit(previsores_treinamento, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data=(previsores_teste, classe_dummy_teste))

#Com redução de dimensionalidade
c2 = Sequential()
c2.add(Dense(units = 21, activation = 'relu', input_dim = previsores_treinamento_codificados.shape[1]))
c2.add(Dense(units = 21, activation = 'relu'))
c2.add(Dense(units = 10, activation = 'softmax'))
c2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
c2.fit(previsores_treinamento_codificados, classe_dummy_treinamento, batch_size = 256,
       epochs = 100, validation_data=(previsores_teste_codificados, classe_dummy_teste))

'''Com redução foi bem mais rápido com perda de 3% de precisão (que pode ser
diminuida case aumente as epochs e/ou aumente a dimensão das imagens). 
Pode ser uma boa aplicação para lugares que precisam treinar redes várias vezes'''