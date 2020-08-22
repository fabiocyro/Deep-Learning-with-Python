import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import Dense, InputLayer, Reshape, Flatten
from keras.models import Sequential
from keras.regularizers import L1L2 #Evita overfitting

#Nesse exemplo tivemos que instalar keras_adversarial e só funciona com a versão 2.1.2 do keras.
# pip install keras == 2.1.2
# pip install --upgrade keras para voltar pra versão atual

'''Caros colegas, caso alguém se depare com problemas na instalação do GAN (mesmo seguindo os passos descritos pelo instrutor), aqui vai um alerta:

Caso esteja usando Tensorflow como backend e surja o seguinte erro ModuleNotFoundError: No module named 'tensorflow.contrib'.

As versões mais recentes do Tensorflow (1.5 e atualmente 2.0) não possuem um recurso utilizado pelo keras_adversarial (o módulo tensorflow.contrib). Para isso, é necessário remover o Tensorflow (pip uninstall tensorflow) e instalar a versão 1.14 (pip install tensorflow==1.14).

Atenção! Isso vai gerar código de aviso em alguns casos. Retorne à versão mais atual assim que terminar de utilizar o GAN.

Fonte: https://github.com/tensorflow/tensorflow/issues/30794'''

from keras_adversarial import AdversarialModel, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, normal_latent_sampling 

(previsores_treinamento, _), (_, _) = mnist.load_data()

previsores_treinamento = previsores_treinamento.astype("float32") / 255

#Gerador

gerador = Sequential()

gerador.add(Dense(units = 500, input_dim = 100, activation = 'relu',
                  kernel_regularizer= (L1L2(1e-5, 1e-5))))

'''a formula entradas + saidas / 2 n funciona bem com GAN, 500 foi de teste e erro
input_dim também, você que escolhe'''

gerador.add(Dense(units = 500, input_dim = 100, activation = 'relu',
                  kernel_regularizer= (L1L2(1e-5, 1e-5))))

unidade_saida = previsores_treinamento.shape[1] * previsores_treinamento.shape[2]
gerador.add(Dense(units = unidade_saida, activation = 'sigmoid', 
                  kernel_regularizer=(L1L2(1e-5, 1e-5))))

gerador.add(Reshape((28,28))) #transformar a saida de 784 em imagens

#Discriminador
discriminador = Sequential()
discriminador.add(InputLayer(input_shape = (28,28)))
discriminador.add(Flatten()) #Necessario converter para vetor mais uma vez para aplicar na ANN
discriminador.add(Dense(units = 500, activation = 'relu',
                        kernel_regularizer=(L1L2(1e-5, 1e-5))))
discriminador.add(Dense(units = 500, activation = 'relu',
                        kernel_regularizer=(L1L2(1e-5, 1e-5))))
discriminador.add(Dense(units = 1, activation = 'sigmoid',
                        kernel_regularizer=(L1L2(1e-5, 1e-5))))

#Construindo efetivamente a GAN

gan = simple_gan(gerador, discriminador, normal_latent_sampling((100,)))
modelo = AdversarialModel(base_model = gan,
                          player_params = [gerador.trainable_weights, 
                                           discriminador.trainable_weights])
modelo.adversarial_compile(adversarial_optimizer = AdversarialOptimizerSimultaneous(),
                           player_optimizers = ['adam', 'adam'],
                           loss = 'binary_crossentropy')
modelo.fit(x = previsores_treinamento, y = gan_targets(60000), epochs = 100, batch_size = 256)
#primeiro foi mostrado pra epochs = 1, e mostrou que o que gerou não foi muito claro, com 100 já se obtém um resultado melhor

amostras = np.random.normal(size = (10,100))

previsao = gerador.predict(amostras)

for i in range(previsao.shape[0]):
    plt.imshow(previsao[i, :], cmap = 'gray')
    plt.show()    