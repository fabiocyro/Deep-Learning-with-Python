import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import cifar10
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense


(previsores_treinamento, _), (previsores_teste, _) = cifar10.load_data()



previsores_treinamento = previsores_treinamento.astype("float32") / 255 #normalização
previsores_teste = previsores_teste.astype("float32") / 255


previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento)), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape((len(previsores_teste)), np.prod(previsores_teste.shape[1:]))

#3072 - 1536 - 768 - 1536 - 3072
autoencoder = Sequential()

#encode
autoencoder.add(Dense(units = 1536, activation = 'relu', input_dim = previsores_treinamento.shape[1]))
autoencoder.add(Dense(units = 768, activation = 'relu'))

#decode
autoencoder.add(Dense(units = 1536, activation = 'relu'))
autoencoder.add(Dense(units = 3072, activation = 'sigmoid'))

autoencoder.summary()

autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                    metrics = ['accuracy'])

autoencoder.fit(previsores_treinamento, previsores_treinamento,
                epochs = 50, batch_size = 256,
                validation_data = (previsores_teste, previsores_teste))


dimensao_original = Input(shape = (previsores_treinamento.shape[1],))
camada_encoder1 = autoencoder.layers[0]
camada_encoder2 = autoencoder.layers[1]


encoder = Model(dimensao_original,
                camada_encoder2(camada_encoder1(dimensao_original)))

encoder.summary()

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)

#visualização

numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)
plt.figure(figsize = (18, 18))

for i, indice_imagens in enumerate(imagens_teste):
    #imagens originais
    eixo = plt.subplot(10, 10, i + 1)
    plt.imshow(previsores_teste[indice_imagens].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())

    #imagens codificadas
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagens].reshape(16,16,3))
    plt.xticks(())
    plt.yticks(())

    #imagens decodificadas
    eixo = plt.subplot(10, 10, i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagens].reshape(32,32,3))
    plt.xticks(())
    plt.yticks(())