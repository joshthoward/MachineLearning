import sys
import numpy as np
import matplotlib.pyplot as plot

from scipy.linalg import eig
from util import *
from PCA import PCA

# Linear Discriminant Analysis
def LDA(train, test, ShowPlot=0, ShowImage=0, Reconstruct=1):
    
    C = train[:,0]
    D = train[:,1:]
    n = np.sqrt(D.shape[1])
    
    label = test[:,0]
    tests = test[:,1:]

    # Calculate class means and covariance matrices
    mean = np.array([np.mean(D[C==0], axis=0), np.mean(D[C==1], axis=0)])
    diff = mean[0] - mean[1]
    covB = np.outer(diff,diff)
    covW = np.cov(D[C==0].T) + np.cov(D[C==1].T)

    # Calculate eigendecomposition
    Eval, Evec = eig(covB,covW)
    Sort = Eval.argsort()[::-1]
    Eval = Eval[Sort]
    Evec = np.real(Evec[:,Sort])

    # Determine what to show
    if (ShowImage == 1):
        imgplot1 = plot.imshow(Evec[0].reshape(n,n))
        plot.show()
        imgplot2 = plot.imshow(Evec[19].reshape(n,n))
        plot.show()
    
    if (ShowPlot == 1):
        Plot_Eval(np.abs(Eval))

    # Project data and do assignments
    W = Evec[:,0]
    P = tests.dot(W.T)

    if (Reconstruct == 1):
        Pinv = W.T/(W.T.dot(W))
        Dnew = D.dot(W.dot(Pinv))
        error = Dnew[0] - D[0]
        print 'Relative Error: {0}'.format(np.linalg.norm(error)/np.linalg.norm(D[0]))
        if (int(n) == n):
            imgplot3 = plot.imshow(error.reshape(n,n))
            plot.show()
    
    Assignment = []
    for i in range(test.shape[0]):
        dist0 = np.linalg.norm(P[i] - W.dot(mean[0]))
        dist1 = np.linalg.norm(P[i] - W.dot(mean[1]))
        if (dist0 < dist1):
            Assignment.append(0)
        else:
            Assignment.append(1)

    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for i in range(test.shape[0]):
        if (Assignment[i] == label[i]):
            if (label[i] == 1):
                TP += 1
            else:
                TN += 1
        else:
            if (label[i] == 1):
                FP += 1
            else:
                FN += 1

    print np.array([[TP,FP],[FN,TN]])


def main(argv):
    
    if (argv[0].upper() == 'WINE'):
        
        
        D = SelectClass(D, 1, 2)
        
        if (argv[1] == '10-fold'):
            
            D = D[np.random.randint(len(D),size=len(D))]
            inc = int(np.floor(D.shape[0]/10))
            for i in range(0, D.shape[0], inc):
                index = np.arange(i,i+inc)
                train = D[index]
                test = np.delete(D,index,axis=0)
                LDA(train, test)
    
        else:
            
            count = int(argv[1])
            C0 = D[D[:,0]==0]
            C1 = D[D[:,0]==1]
            
            I0 = np.arange(C0.shape[0])
            I1 = np.arange(C1.shape[0])
            np.random.shuffle(I0)
            np.random.shuffle(I1)
            
            train = np.concatenate((C0[I0[:count]],C1[I1[:count]]), axis=0)
            test  = np.concatenate((C0[I0[count:]],C1[I1[count:]]), axis=0)
            LDA(train, test)

    if (argv[0].upper() == 'MNIST'):
    
        train = np.genfromtxt('train.csv', delimiter=',')
        train = SelectClass(train.T, int(argv[1]), int(argv[2]), 'MNIST')
        
        test = np.genfromtxt('test.csv', delimiter=',')
        test = SelectClass(test.T, int(argv[1]), int(argv[2]), 'MNIST')
        
        LDA(train, test)

if __name__ == '__main__':
    main(sys.argv[1:])