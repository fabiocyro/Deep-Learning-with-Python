from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 3)

base = np.array([[0,1,1,1,0,1],
                 [1,1,0,1,1,1],
                 [0,1,0,1,0,1],
                 [0,1,1,1,0,1],
                 [1,1,0,1,0,1],
                 [1,1,0,1,1,1]])

filmes = ['Freddy x Jason', 'O Ultimato Bourne',
          'Star Trek', 'O Exterminador do Futuro',
          'Norbit', 'Star Wars']

rbm.train(base, max_epochs = 5000)
rbm.weights

leonardo = np.array([[0,1,0,1,0,1]])

resultado = rbm.run_visible(leonardo)

recomendacao = rbm.run_hidden(resultado)

for i in range(len(leonardo[0])):
    if leonardo[0, i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])