import numpy as np
import bottleneck as bn
import sys

from scipy.spatial.distance import cdist

# Performs Data Whitening
def whiten(D):
    
    C = np.cov(D.T)
    V, E = np.linalg.eigh(C)
    for i in range(V.shape[0]):
        if V[i] != 0:
           V[i]  = 1/np.sqrt(np.abs(V[i]))

    return np.diag(V).dot(E.T)

# Dynamic Time Warping (for time series KNN)


# k-Nearest Neighbors
class KNN:

    def __init__(self, k, trainDat, trainLab):
        
        self.k = k
        self.X = trainDat
        self.y = trainLab

    def predict(self, testDat):

        pred = np.zeros(testDat.shape[0])
            
        for j,t in enumerate(testDat):
            
            distances = cdist(t[np.newaxis,:], self.X, 'euclidean').ravel()
            
            index = bn.argpartsort(distances, n=self.k)
            label = self.y[index[:self.k]]
            votes = {}

            for i in label:
                if i in votes:
                    votes[i] += 1
                else:
                    votes[i]  = 1

            pred[j] = max(votes.iteritems(), key=operator.itemgetter(1))[0]
    
        return pred
