''' EXEMPLO DE VALIDAÇÃO CRUZADA 

Alguns comentarios: 

Exemplo feito após breas_cancer_simples: Significa que funções repetidas ou
desnecessárias de comentários foram comentadas apenas no outro arquivo    
k = 10 é o mais comum usado na comunidade científica
Validação é o mais comumente utilizado pois evita perda de bons previsores
quand se faz a divisão entre treinamento e teste.
'''
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')



def criarRede():
    
    classificador = Sequential() 
    classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform", input_dim = 30)) 
    classificador.add(Dropout(0.2)) #Zera alguns valores da entrada, de modo a eviter overfitting (valores muito altos podem causar underfitting)
    classificador.add(Dense(units = 16, activation = "relu",
                        kernel_initializer = "random_uniform")) 
    classificador.add(Dropout(0.2))

    classificador.add(Dense(units = 1, activation = "sigmoid"))


    otimizador = keras.optimizers.Adam(lr = 0.001, decay = 0.0001, clipvalue = 0.5) 


    classificador.compile(optimizer = otimizador, loss = "binary_crossentropy",
                     metrics = ["binary_accuracy"])
    return classificador

classificador = KerasClassifier(build_fn = criarRede,
                                 epochs = 100,
                                 batch_size = 10)

resultados = cross_val_score(classificador,
                             X = previsores, y = classes,
                             cv = 10, scoring = 'accuracy')

media = resultados.mean()
desvio = resultados.std() #Quanto MAIOR, mais a possibilidade de overfitting