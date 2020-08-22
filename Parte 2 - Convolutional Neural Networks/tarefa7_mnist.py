from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
import numpy as np

#Carregar a base de dados
(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

#Pre processamento
previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0], 28, 28, 1)
previsores_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

previsores_treinamento = previsores_treinamento.astype('float32')
previsores_teste = previsores_teste.astype('float32')

previsores_treinamento /= 255
previsores_teste /= 255

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#Criar a rede Neural
classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape = (28,28,1),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10, activation = 'softmax'))

classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                      metric = ['accuracy'])

classificador.fit(previsores_treinamento, classe_treinamento,
                  batch_size = 128, epochs = 5,
                  validation_data = (previsores_teste, classe_teste))

resultados = classificador.evaluate(previsores_teste, classe_teste)

#Classificar somente uma imagem
plt.imshow(X_teste[1], cmap = 'gray')
plt.title('Classe: ' + str(y_teste[1]))

imagem_teste = X_teste[1].reshape(1, 28, 28, 1)
imagem_teste = imagem_teste.astype('float32')
imagem_teste /= 255

previsao = classificador.predict(imagem_teste)

print("O numero pertence a classe: " + str(np.argmax(previsao)))