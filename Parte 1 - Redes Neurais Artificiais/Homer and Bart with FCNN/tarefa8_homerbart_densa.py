from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

base = pd.read_csv('personagens.csv')

previsores = base.iloc[:, 0:6].values
classes = base.iloc[:, 6].values

labelenconder = LabelEncoder()
classes = labelenconder.fit_transform(classes)
classes_dummy = np_utils.to_categorical(classes)


previsores_treinamento, previsores_teste, classes_treinamento, classes_teste = train_test_split(previsores, classes_dummy, test_size = 0.25)


classificador = Sequential()

classificador.add(Dense(units = 4, activation = 'relu', input_dim = 6))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 4, activation = 'relu'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 2, activation = 'sigmoid'))

classificador.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
classificador.fit(previsores_treinamento, classes_treinamento, batch_size = 10, epochs = 1000)

resultado = classificador.evaluate(previsores_teste, classes_teste)

previsoes = classificador.predict(previsores_teste)
previsoes = (previsoes > 0.5)