import numpy as np
import time

from sklearn.preprocessing import LabelEncoder

class SVM():

    def __init__(self, C=1, max_iter=50, tol=0.001):
        
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.LE = LabelEncoder()

    def fit(self, X, y):
        
        m,n = X.shape
        y = self.LE.fit_transform(y)
        k = len(self.LE.classes_)
        
        self.weight = np.zeros((n,k))
        
        norms = np.linalg.norm(X,axis=1)
        alpha = np.zeros((k,m))
        ratio = 1
        
        # Run iterative updates
        iter = -1
        while (iter < self.max_iter and ratio > self.tol):
            iter += 1
            viol = 0

            for i in range(m):

                pos = np.array([index == y[i] for index in range(k)])
                neg = np.logical_not(pos)
                
                grad = np.dot(X[i],self.weight)
                grad[neg] += 1
               
                a = np.logical_or(neg,alpha[:,i] < self.C)
                b = np.logical_or(pos,alpha[:,i] < 0)

                v = grad.max() - grad[np.logical_and(a,b)].min()
                viol += v

                if 1e-8 <= v:
                    
                    coef = -alpha[:,i]
                    coef[pos] += self.C
                    beta = norms[i]*coef+grad/norms[i]
        
                    dec = np.argsort(beta)[::-1]
                    acc = np.cumsum(beta[dec])-self.C*norms[i]
                    index = np.arange(1,k+1)
                    j = beta[dec] > (acc/index)
                    beta -= acc[j][-1]/index[j][-1]
        
                    update = coef-np.maximum(beta/norms[i],0)
            
                    self.weight += update * X[i][:,np.newaxis]
                    alpha[:, i] += update

            if iter == 0:
                init = viol

            ratio = viol/init

    def predict(self, X):
        return self.LE.inverse_transform(X.dot(self.weight).argmax(axis=1))


def main():
    
    data = np.loadtxt('HAR.txt', delimiter=',')

    K = 10
    
    TrTime = np.zeros(K)
    TeTime = np.zeros(K)
    Accuracy = np.zeros(K)

    for k in range(K):
        
        trainDat = np.array([x for i, x in enumerate(data[:,1:]) if i % K != k])
        trainLab = np.array([x for i, x in enumerate(data[:,0 ]) if i % K != k])
        testDat  = np.array([x for i, x in enumerate(data[:,1:]) if i % K == k])
        testLab  = np.array([x for i, x in enumerate(data[:,0 ]) if i % K == k])
        
        clf = SVM()
        
        start = time.time()
        clf.fit(trainDat, trainLab)
        TrTime[k] = time.time()-start

        start = time.time()
        predLab = clf.predict(testDat)
        TeTime = time.time()-start

        Accuracy[k] = np.average(predLab==testLab)

    print('Accuracy: ', np.average(Accuracy))
    print('Training Time: ', np.average(TrTime))
    print('Testing Tiime: ', np.average(TeTime))

if __name__ == '__main__':
    main()