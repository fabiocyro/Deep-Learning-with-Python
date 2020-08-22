from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6, num_hidden = 2)

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,1,0,1]])

filmes = ['A bruxa','Invocação do mal', 'O chamado', 'Se beber não case',
          'Gente grande', 'American pie']



rbm.train(base, max_epochs = 5000)
rbm.weights
'''com rbm.weights vc pode ver a matriz dos pesos obviamente, mas o que é
interessante de se ver é que:
    a primeira linha representa o bias,
    as linhas seguintes são os filmes, onde o primeiro valor é o filme (id de
    certa forma) e os outros valores representam qual neurônio foi escolhido.
    achando assim um padrão'''


usuario1 = np.array([[1,1,0,1,0,0]])
usuario2 = np.array([[0,0,0,1,1,0]])



resultado1 = rbm.run_visible(usuario1)
resultado2 = rbm.run_visible(usuario2)


recomendacao1 = rbm.run_hidden(resultado1)
recomendacao2 = rbm.run_hidden(resultado2)

for i in range(len(usuario1[0])):
    if usuario1[0, i] == 0 and recomendacao1[0,i] == 1:
        print(filmes[i])


for i in range(len(usuario2[0])):
    if usuario2[0, i] == 0 and recomendacao2[0,i] == 1:
        print(filmes[i])
    