from minisom import MiniSom #biblioteca para criar os mapas
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot #para visualização dos dados

base = pd.read_csv("wines.csv")

X = base.iloc[:, 1:14].values
y = base.iloc[:, 0].values

#como os valores de X não estão normalizados precisamos realizar os códigos seguintes
normalizador = MinMaxScaler(feature_range= (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(8, 8, input_len = X.shape[1], sigma = 1.0, learning_rate = 0.5, random_seed = 2)

'''Para saber o tamanho do mapa usa-se a regra de 5sqr(N), como temos 178 registros:
    5 x sqr(178) = 65,65, aproximado a 64, que dá uma matrix 8x8. Sigma = raio do BMU
    (best matching unit)'''

som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100) #100 é suficiente para a maioria dos casos

som._weights
'''Todos esses valores mostrados pelos pesos representam os novos pontos criados
de modo a ajudar na criação do mapa'''
som._activation_map #Aqui se visualiza os valores do mapa em si
q = som.activation_response(X) #Mostra uma matriz que diz quantas vezes cada neuronios foi eleito BMU

pcolor(som.distance_map().T)
colorbar()
#MID = Mean inter-neuron distance, o código acima ver a semelhança de um determinado neuronio com os seus vizinhos (usado para métrica)

w = som.winner(X[1]) #mostra qual foi o neurônio BMU para a entrada em questão
markers = ["o", "s", "D"]
color = ['r', 'g', 'b']

'''Mudança necessária para colocar os marcadores, pois originalmente as classes são 
1, 2 e 3, só que os indices dos marcadores são 0, 1 e 2, isso geraria conflito'''
#y[y==1] = 0
#y[y==2] = 1 comentar essas linhas após criar o for loop abaixo
#y[y==3] = 2

for i, x in enumerate(X):
    #print(i) indice
    #print(x) valores
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor = "None",
         markersize = 10, markeredgecolor = color[y[i]], markeredgewidth = 2)