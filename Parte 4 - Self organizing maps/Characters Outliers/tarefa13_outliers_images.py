from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from pylab import plot, colorbar, pcolor

base = pd.read_csv("personagens.csv")


X = base.iloc[:, 0:6].values
y = base.iloc[:, 6].values

normalizador = MinMaxScaler(feature_range = (0,1))
X = normalizador.fit_transform(X)

som = MiniSom(x = 9, y = 9, input_len = X.shape[1])
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


y[y == "Bart"] = 0
y[y == "Homer"] = 1

pcolor(som.distance_map().T)
colorbar()


markers = ['o', 's']
color = ['red', 'green']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5, w[1] + 0.5, markers[y[i]],
         markerfacecolor = 'None', markersize = 10,
         markeredgecolor = color[y[i]], markeredgewidth = 2)


#mapeamento = som.win_map(X) #esse codigo mostra quais valores escolheram quais no gráfico

#suspeitos = np.concatenate((mapeamento[(11, 13)], mapeamento[(9, 13)]), axis = 0)
#suspeitos = normalizador.inverse_transform(suspeitos)

classe = []

for i in range(len(base)):
    for j in range(len(suspeitos)):
        if base.iloc[i, 0] == int(round(suspeitos[j, 0])):
            classe.append(base.iloc[i, 4])

classe = np.asarray(classe)

suspeitos_final = np.column_stack((suspeitos, classe))
suspeitos_final = suspeitos_final[suspeitos_final[:, 4].argsort()]

