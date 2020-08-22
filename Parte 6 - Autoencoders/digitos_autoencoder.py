import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense


(previsores_treinamento, _), (previsores_teste, _) = mnist.load_data()

''' o _ seria inicialmente para colocar classe_treinamento e classe_teste
respectivamente, porém como estamos estudando autoencoders, vamos deixar ele
criar as classificações automaticamente'''

previsores_treinamento = previsores_treinamento.astype("float32") / 255 #normalização
previsores_teste = previsores_teste.astype("float32") / 255
#pode-se usar MinMaxScaler

previsores_treinamento = previsores_treinamento.reshape((len(previsores_treinamento)), np.prod(previsores_treinamento.shape[1:]))
previsores_teste = previsores_teste.reshape((len(previsores_teste)), np.prod(previsores_teste.shape[1:]))



autoencoder = Sequential()
autoencoder.add(Dense(units = 32, activation = 'relu', input_dim = previsores_treinamento.shape[1]))
autoencoder.add(Dense(previsores_treinamento.shape[1], activation = "sigmoid"))
autoencoder.summary()
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
autoencoder.fit(previsores_treinamento, previsores_treinamento, 
                epochs = 50, batch_size = 256, #Number of epochs é bem importante para AE, quanto mais melhor
                validation_data=(previsores_teste, previsores_teste))


dimensao_original = Input(shape = previsores_treinamento.shape[1],)
camada_encoder = autoencoder.layers[0]
encoder = Model(dimensao_original, camada_encoder(dimensao_original))
encoder.summary()

imagens_codificadas = encoder.predict(previsores_teste)
imagens_decodificadas = autoencoder.predict(previsores_teste)


numero_imagens = 10
imagens_teste = np.random.randint(previsores_teste.shape[0], size = numero_imagens)

plt.figure(figsize=(18,18))

for i, indice_imagem in enumerate(imagens_teste):
    
    #imagem original
    plt.subplot(10,10,i + 1)
    plt.imshow(previsores_teste[indice_imagem].reshape(28,28))
    plt.xticks(()) #retirar os numeros do eixo
    plt.yticks(())
    
    #imagem codificadas
    plt.subplot(10,10,i + 1 + numero_imagens)
    plt.imshow(imagens_codificadas[indice_imagem].reshape(8,4))
    plt.xticks(()) #retirar os numeros do eixo
    plt.yticks(())
    
    #imagem decodificadas
    plt.subplot(10,10,i + 1 + numero_imagens * 2)
    plt.imshow(imagens_decodificadas[indice_imagem].reshape(28,28))
    plt.xticks(()) #retirar os numeros do eixo
    plt.yticks(())