import sys
import numpy as np
import matplotlib.pyplot as plot

from util import *

# Principle Component Analysis
def PCA(D, ShowPlot=0, ShowImage=0, Reconstruct=0):
    
    # Calculate components
    m = D.shape[0]
    n = np.sqrt(D.shape[1])
    D = D - np.outer(np.ones(m,),np.mean(D, axis=0))
    D = D/np.sqrt(m-1)
    U, S, V = np.linalg.svd(D, full_matrices=True)
    W = V[0:6].T
    
    # Determine what to show
    if (ShowPlot == 1):
        Plot_Eval(np.power(S,2))
    
    if (ShowImage == 1):
        imgplot1 = plot.imshow(V[0].reshape(n,n))
        imgplot2 = plot.imshow(V[19].reshape(n,n))
        plot.show()

    if (Reconstruct == 1):
        Pinv = np.linalg.inv(W.T.dot(W)).dot(W.T)
        Dnew = D.dot(W.dot(Pinv))
        error = Dnew[0] - D[0]
        print 'Relative Error: {0}'.format(np.linalg.norm(error)/np.linalg.norm(D[0]))
        if (int(n) == n):
            imgplot3 = plot.imshow(error.reshape(n,n))
            plot.show()

    # Return matrix of components
    return V

def main(argv):
    
    # Select data set
    if (argv[0].upper() == 'WINE'):
        
        D = np.genfromtxt('wine.data', delimiter=',')
        D = SelectClass(D, 1, 2)
        
        PCA(D[:,1:])

    if (argv[0].upper() == 'MNIST'):
        
        train = np.genfromtxt('train.csv', delimiter=',')
        train = SelectClass(train.T, int(argv[1]), int(argv[2]), 'MNIST')
        
        PCA(train[:,1:])

if __name__ == '__main__':
    main(sys.argv[1:])