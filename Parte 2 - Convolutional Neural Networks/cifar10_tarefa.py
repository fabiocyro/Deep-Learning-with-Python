#Importando os módulos necessários
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import np_utils
from tensorflow.python.keras.layers.normalization import BatchNormalization #importação da 
  

(X_treinamento, y_treinamento) , (X_teste, y_teste) = cifar10.load_data()
plt.imshow(X_treinamento[1], cmap = 'gray')
plt.title('Classe: ' + str(y_treinamento[1]))


previsores_treinamento = X_treinamento.reshape(X_treinamento.shape[0],
                                               32, 32, 3)

previsores_teste = X_teste.reshape(X_teste.shape[0], 32, 32, 3)

#É preciso converter de uint8 para float32, pois no próximo passo precisamos
#de números decimais
previsores_treinamento = previsores_treinamento.astype("float32")

previsores_teste = previsores_teste.astype("float32")

#Converter os pixels para uma escala entre 0 e 1 para facilitar o 
#processamento (normalização)
previsores_treinamento /= 255
previsores_teste /= 255

#Criar as classes dummy, pois como os números vao de 0 a 9 é preciso converter
#para que seja possível classificar, por exemplo:
#Classe 0 = 0 0 0 0 0 0 0 0 0 
#Classe 1 = 1 0 0 0 0 0 0 0 0 e assim sucessivamente

classe_treinamento = np_utils.to_categorical(y_treinamento, 10)
classe_teste = np_utils.to_categorical(y_teste, 10)

#Criando o classificador:
classificador = Sequential()
classificador.add(Conv2D(64, (3,3), input_shape = (32, 32, 3),
                         activation = "relu"))
'''por convenção, recomenda-se começar com 64 filtros ao invés de 32 e depois
ir aumentando, foi-se utilizado uma quantidade menor por motivos de 
aprendizado'''
classificador.add(BatchNormalization()) #Essa camada normaliza os kernels (segunda aula pós criação)
classificador.add(MaxPooling2D(pool_size = (2,2)))
#classificador.add(Flatten()) Dica da segunda aula: Flatten só antes de aplicar a rede em si, se precisar adicionar
#mais convlayers, não usa.

#As quatro linhas a seguir foram também da aula seguinte após criação da rede,
#são feitas para melhorar a qualidade da rede
classificador.add(Conv2D(64, (3,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
'''para se calcular a quantidade de neuronios na camada de entrada, calcula-se
em relação ao tamanho da matriz que se obtem após realizar o pooling. Então,
soma-se com a saída e divide-se por dois, como anteriormente. Essa prática não
é muito comum em ConvNets, em geral começa com 128 e vai aumentando para 256,
512 etc
Da entrada para a camada de convolução reduz-se em 2, de 28x28 foi para 26x26
Após o pooling isso cai pra metade, 13x13. Daí então se calcula'''
classificador.add(Dropout(0.2)) #Também aula seguinte
classificador.add(Dense(units = 128, activation = 'relu')) #Também aula seguinte
classificador.add(Dropout(0.2)) #Também aula seguinte
classificador.add(Dense(units = 10, activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy', optimizer = 'adam',
                      metrics = ['accuracy'])
classificador.fit(previsores_treinamento, classe_treinamento, batch_size = 128,
                  epochs = 5, validation_data = (previsores_teste, classe_teste))
'''epochs poucas pois é um processo demorado'''

resultado = classificador.evaluate(previsores_teste, classe_teste)
