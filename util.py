import matplotlib.pyplot as plot
import matplotlib.image as plimg
import numpy as np

def SelectClass(D, id0, id1, type='none'):

    if (type == 'MNIST'):
        end = D.shape[1]
        C = D[:,end-1]
        D[:,1:] = D[:,1:end]
        D[:,0 ] = C
        
    index = []
    end = D.shape[1]
    for i,j in enumerate(D[:,0]):
        if (j == id0):
            D[i,0] = 0
            index.append(i)
        elif (j == id1):
            D[i,0] = 1
            index.append(i)

    return D[index]

def Plot_Eval(Eval):
    
    if (len(Eval) > 20):
        length = 20
    else:
        length = len(Eval)

    Eval = np.abs(Eval)
    Position = np.arange(length)

    Sum = np.sum(Eval)
    for i in range(len(Eval)):
        Eval[i] = Eval[i]/Sum

    Eval = 100*Eval[Position]
    Position = Position + np.ones(length)

    plot.bar(Position, Eval, align='center', color='r', alpha=0.6)
    plot.xticks(Position)
    plot.xlabel('Eigenvalue')
    plot.ylabel('Cumulative Sum (%)')
    plot.show()