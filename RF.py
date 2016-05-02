import sys
import numpy as np
import scipy as sc

from sklearn.tree import DecisionTreeClassifier

class Forest:
    
    def __init__( self, dep, num):
        
        self.num = num
        self.dep = dep
        self.trees = []
    
    def Fit(self, X, y):
        
        self.trees = []
        samples = np.round(np.sqrt(X.shape[1]));

        for i in range(self.num):
            index = np.random.randint(X.shape[0], size=samples)
            clf = DecisionTreeClassifier(max_depth=self.dep)
            clf.fit(X[index],y[index])
            self.trees.append(clf)

    def OutOfBag(self, X, y):

        self.trees = []
        samples = np.round(np.sqrt(X.shape[1]));
        
        N = X.shape[0]
        
        indices = []
        for i in range(self.num):
            index = list(np.random.randint(N, size=samples))
            temp = []
            for i in range(N):
                if i not in index:
                    temp.append(i)
            
            indices.append(temp)
            clf = DecisionTreeClassifier(max_depth=self.dep)
            clf.fit(X[index],y[index])
            self.trees.append(clf)
    
        votes = np.zeros((N,self.num))
        for i in range(self.num):
            votes[indices[i],i] = self.trees[i].predict(X[indices[i]])

        return sc.stats.mode(votes, axis=1)

    def Predict(self, X):

        votes = np.zeros((X.shape[0],self.num))
        for i in range(self.num):
            votes[:,i] = self.trees[i].predict(X)
        
        return sc.stats.mode(votes, axis=1)
