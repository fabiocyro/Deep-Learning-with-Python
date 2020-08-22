from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.python.keras.layers.normalization import BatchNormalization #importação da 
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image

classificador = Sequential()

classificador.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))

classificador.add(Flatten())

classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 1, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy',
                      metrics = ['accuracy'])

gerador_treinamento = ImageDataGenerator(rescale = 1./255,
                                         rotation_range = 7,
                                         horizontal_flip = True,
                                         shear_range = 0.2,
                                         height_shift_range = 0.07,
                                         zoom_range = 0.2)
'''ImageDataGenerator é interessante usar, mesmo que não sendo necessário fazer
augumentation, pois a função rescale normaliza as imagens, sem ser
necessário carregar as images, transformar pra vetor numpy etc etc etc'''

gerador_teste = ImageDataGenerator(rescale = 1./255)

base_treinamento = gerador_treinamento.flow_from_directory('dataset/training_set',
                                                           target_size = (64, 64),
                                                           batch_size = 32,
                                                           class_mode = 'binary') 

base_teste = gerador_teste.flow_from_directory('dataset/test_set',
                                               target_size = (64,64),
                                               batch_size = 32,
                                               class_mode = 'binary')  

classificador.fit(base_treinamento, steps_per_epoch = 4000 / 32,
                  epochs = 5, validation_data = base_teste,
                  validation_steps = 1000 / 32)

'''Lembrar que o recomendável pros steps_per_epoch é não dividir pelo batch size
e sim usar todas as imagens, porém exige uma maior capacidade de processamento
(sem esquecer obviamente de aumentar o número de epochs.'''

#Aula seguinte mostrando como classificar uma imagem com a rede neural treinada

imagem_teste = image.load_img('dataset/test_set/gato/cat.3500.jpg', 
                              target_size = (64, 64))

imagem_teste = image.img_to_array(imagem_teste) #Transforma em um array (target_size, numero de canais)
imagem_teste /= 255
imagem_teste = np.expand_dims(imagem_teste, axis = 0) #Converte para o formato que o tensorflow reconhece (1, 64, 64, 3), onde 1 é a quantidade de linhas

previsao = classificador.predict(imagem_teste)

base_treinamento.class_indices