import numpy as np
import pandas as pd
from keras.models import model_from_json

arquivo = open('classificador_json', 'r')
estrutura_rede = arquivo.read()
arquivo.close() 

classificador = model_from_json(estrutura_rede)
classificador.load_weights('classificador_iris.h5')

# iris setosa     1 0 0
# iris virginica  0 1 0
# iris versicolor 0 0 1

novo = np.array([[13.4, 12.2, 15.2, 9]])

previsao = classificador.predict(novo)
previsao = (previsao > 0.5)

if previsao[0,0]:
    print("Iris Setosa")
elif previsao[0,1]:
    print("Iris Virgínica")
else:
    print("Iris Versicolor")

#Para fazer uma avaliação com uma base de dados de teste:

'''previsores = pd.read_csv('entradas_breast.csv')
classes = pd.read_csv('saidas_breast.csv')

classificador.compile(loss = "binary_crossentropy", optimizer = 'adam', 
                      metrics = ['binary_accuracy'])

resultado = classificador.evaluate(previsores, classes)'''
