import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils #para organizar as classes em (1, 0, 0), (0, 1, 0), (0, 0, 1)

base = pd.read_csv('iris.csv') #esse arquivo tem os previsores e as classes tudo junto
previsores = base.iloc[:, 0:4].values #pegando somento os previsores (.values deixa em numpy)
classes = base.iloc[:, 4].values #pegando só as classes

from sklearn.preprocessing import LabelEncoder #para transformar os valores das classes de iris-setosa para 0 por exemplo
labelenconder = LabelEncoder()
classes = labelenconder.fit_transform(classes)
classes_dummy = np_utils.to_categorical(classes)

from sklearn.model_selection import train_test_split 
previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes_dummy, test_size = 0.25) #separar entre treinamento e teste

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dense(units = 3, activation = 'softmax')) #Sempre lembrar que para mais de uma saída, a função de ativação melhor é a softmax
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(previsores_treinamento, classes_treinamento, batch_size = 10,
                  epochs = 1000)

''' NOTA MENTAL: É preciso fazer um tratamento na variável classes_treinamento, pois inicialmente
gera um erro, pois as saídas estavam em formato string. A importação sklearn.preprocessing é onde se solucina isso
(basicamente transforma para números, tipo, a primeira classificação ao invés de ser iris-blabla vira 0, a próxima 1, etc);
Porém ainda é necessário realizar outras mudanças, pois como se possui três saídas o primeiro valor tem que ser (1, 0, 0) por exemplo,
e fazendo apenas a conversão, vai retornar 0, 1 ou 2, e os outros neurônios da camada de saída ficarão vazios, o que não pode.
Essa correção é feita com a importção keras.utils import np_utils 
Esse comentário é apenas pra vc se lembrar de como foi a aula.'''

resultado = classificador.evaluate(previsores_teste, classes_teste) #evaluate função do keras que aplicar os testes e retorna a loss function e a accuracy
previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)

'''Importações necessárias a seguir para a matriz de confusão: A matriz de confusão não reconhece o formato (1,0,0), (0,1,0) e (0,0,1),
as linhas de código a seguir pegam o índice de maior valor e retornam somente ele nas novas variáveis'''
import numpy as np
classes_teste2 = [np.argmax(t) for t in classes_teste]
previsoes2 =[np.argmax(t) for t in previsoes]

#matriz de confusão
from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(previsoes2, classes_teste2) 