from minisom import MiniSom
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar, plot

base1 = pd.read_csv("entradas_breast.csv")
X = base1.iloc[:, 0:30].values

base2 = pd.read_csv("saidas_breast.csv")
y = base2.iloc[:, 0].values

#como os valores de X não estão normalizados precisamos realizar os códigos seguintes
normalizador = MinMaxScaler(feature_range= (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(11, 11, input_len = X.shape[1], sigma = 2.5, learning_rate = 0.3, random_seed = 0)

som.random_weights_init(X)
som.train_random(data = X, num_iteration = 1000) #100 é suficiente para a maioria dos casos

pcolor(som.distance_map().T)
colorbar()

markers = ["o", "s"]
color = ['r', 'g']

for i, x in enumerate(X):
    #print(i) indice
    #print(x) valores
    w = som.winner(x)
    #print(w)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]], markerfacecolor = "None",
         markersize = 10, markeredgecolor = color[y[i]], markeredgewidth = 2)