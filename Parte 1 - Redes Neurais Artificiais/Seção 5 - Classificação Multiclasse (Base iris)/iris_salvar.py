import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelenconder = LabelEncoder()
classe = labelenconder.fit_transform(classe)
#classe_dummy = np_utils.to_categorical(classe)

classificador = Sequential()
classificador.add(Dense(units = 4, activation = 'relu', input_dim = 4))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 3, activation = 'softmax')) #Sempre lembrar que para mais de uma saída, a função de ativação melhor é a softmax
classificador.compile(optimizer = 'sgd', loss = 'sparse_categorical_crossentropy',
                      metrics = ['accuracy'])

classificador.fit(previsores, classe, 20, 900)

classificador_json = classificador.to_json()

with open ('classificador_json', 'w') as json_file:
    json_file.write(classificador_json)
classificador.save_weights('classificador_iris.h5')
