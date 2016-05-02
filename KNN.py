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
def DTW(self, a, b):

    metric = lambda x,y : np.abs(np.subtract(x,y))
        
    m = a.shape[0]
    n = b.shape[0]
    D = np.iinfo(np.int32).max * np.ones((m,n))
        
    D[:,0] = np.cumsum(metric(b[0],a))
    D[0,:] = np.cumsum(metric(a[0],b))

    for i in range(1,m):
        for j in range(max(1,i-self.w),min(n,i+self.w)):
            D[i,j] = min(D[i-1,j-1],D[i,j-1],D[i-1,j]) + metric(a[i],b[j])

    return D[-1,-1]

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
