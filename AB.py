import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier

# Multiclass Adaboost Implementation (1 vs. All)
class AdaBoost:

    def __init__(self, dep, itr, names):
        
        self.itr = itr
        self.dep = dep
        self.names = names
        self.C = []
        self.A = []
    
    def Train(self, X, y):
        
        N = X.shape[0]
        New = np.zeros(N)

        for i in self.names:
            
            A = []
            C = []
            
            weight = np.ones(N)/N
            New[np.where(y == i)[0]] =  1
            New[np.where(y != i)[0]] = -1
            
            for j in range(self.itr):
                
                # Input a generic sklearn classifier
                clf = DecisionTreeClassifier(max_depth = self.dep)
                clf.fit(X, New, sample_weight = weight)
                Pre = clf.predict(X)
                err = weight.dot((New != Pre).astype(int))

                if (err != 0):
                    A.append(.5*np.log((1-err)/err))
                else:
                    A.append(1)

                C.append(clf)
                weight *= np.exp(-A[j]*New*Pre)
                weconight = weight/np.sum(weight)

            self.C.append(C)
            self.A.append(A)
            
    def Test(self, X):

        N = X.shape[0]
        M = self.names.shape[0]
        
        votes = np.zeros([N,M])
        for i in range(len(self.names)):
            for j in range(self.itr):
                votes[:,i] += self.A[i][j]*self.C[i][j].predict(X)
        
        return self.names[np.argmax(votes,axis=1)]
