import numpy as np
import bottleneck as bn
import sys
import time

from scipy.stats import mode

'''
Activity labels are 
1: WALKING
2: WALKING_UPSTAIRS
3: WALKING_DOWNSTAIRS
4: SITTING
5: STANDING
6: LAYING
'''

class KNN:
    
    def __init__(self, k, w, trainDat, trainLab):
        
        self.w = w
        self.k = k
        self.X = trainDat
        self.y = trainLab

    # Dynamic Time Warp
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

    # Classifier
    def predict(self, Te):

        Tr = self.X
        
        m = Te.shape[0]
        n = Tr.shape[0]
        P = np.zeros(m)
        D = np.zeros((m,n))
        
        for i in range(m):
            start = time.time()
            for j in range(n):
                D[i,j] = self.DTW(Te[i],Tr[j])
            index = bn.argpartsort(D[i], n=self.k)[:self.k]
            P[i] = mode(self.y[index])[0][0]
            print(time.time()-start)
        return P

def main(input):

    data = np.loadtxt('HAR.txt', delimiter=',')
    
    results = []

    K = 10
    for k in range(1):
        
        trainDat = np.array([x for i, x in enumerate(data[:,1:]) if i % K != k])
        trainLab = np.array([x for i, x in enumerate(data[:,0 ]) if i % K != k])
        testDat  = np.array([x for i, x in enumerate(data[:,1:]) if i % K == k])
        testLab  = np.array([x for i, x in enumerate(data[:,0 ]) if i % K == k])

        nearest_neighbors = KNN(10, 2, trainDat[::10], trainLab[::10])
        predLab = nearest_neighbors.predict(testDat[::10])
        
        print('Accuracy: ', np.average(predLab==testLab[::10])) 
        
if __name__ == '__main__':
    main(sys.argv[1:])






