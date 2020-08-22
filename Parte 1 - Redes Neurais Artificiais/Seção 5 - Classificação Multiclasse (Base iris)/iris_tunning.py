import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import backend as k #O código demorou muito pra executar, isso é suposto de reduzir o tempo


base = pd.read_csv('iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values



def criar_rede(optimizer, kernel_initializer, activation, neurons):
    k.clear_session()
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, input_dim = 4))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = neurons, activation = activation))
    classificador.add(Dropout(0.2))
    classificador.add(Dense(units = 3, activation = 'softmax')) #Sempre lembrar que para mais de uma saída, a função de ativação melhor é a softmax
    classificador.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy',
                          metrics = ['accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criar_rede)
parametros = {'batch_size': [10, 20], 
              'epochs': [900, 1000],
              'optimizer': ['adam', 'sgd'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'tanh'],
              'neurons': [4, 6]}

grid_search = GridSearchCV(estimator = classificador, param_grid = parametros,
                           cv = 10)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_
