import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plot

from sklearn.tree import DecisionTreeClassifier

np.set_printoptions(threshold=np.nan)

def format(D):
    D = D.T
    n = D.shape[1]
    Lab = D[:, n-1]
    Dat = D[:,:n-1]
    return Dat, Lab

def readData(file):
    
    if (file == 'Wine'):
        D = np.genfromtxt('../Wine/wine.data', delimiter=',')
        D = D[np.random.randint(len(D),size=len(D))]
        m = D.shape[0]
        p = round(.75*m)
        
        return D[:p,1:], D[p:,1:], D[:p,0], D[p:,0]
    
    elif (file == 'MNIST'):
        train = np.genfromtxt('../MNIST/train.csv', delimiter=',')
        test = np.genfromtxt('../MNIST/test.csv', delimiter=',')
        trainDat, trainLab = format(train)
        testDat, testLab = format(test)
        
        return trainDat, testDat, trainLab, testLab

    else: sys.exit()

class Forest:
    
    def __init__( self, dep, num):
        
        self.num = num
        self.dep = dep
        self.trees = []
    
    def Train(self, X, y):
        
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

    def Test(self, X):

        votes = np.zeros((X.shape[0],self.num))
        for i in range(self.num):
            votes[:,i] = self.trees[i].predict(X)
        
        return sc.stats.mode(votes, axis=1)

def main(input):
    # Parse command line args
    if (len(input) < 1):
        sys.exit()

    if (len(input) < 3):
        input.append('')
        input.append(1)

    if (input[1].isdigit()):
        input[1] = int(input[1])
    else:
        input[1] = None

    # Read data file
    trainDat, testDat, trainLab, testLab = readData(input[0])

    # Initialize Classifier
    F = Forest(input[1], int(input[2]))
    F.Train(trainDat, trainLab)
    classLab,_ = F.Test(testDat)

    n = classLab.shape[0]
    count = 0
    for i in range(n):
        if (testLab[i] == classLab[i]):
            count += 1

    print count/float(n)

if __name__ == "__main__":
    main(sys.argv[1:])


import csv, sys
import numpy as np

from scipy.stats import mode

from CSE6242HW4Tester import generateSubmissionFile

myname = "Howard-Joshua"

class RandomForest(object):
    
    class DecisionTree(object):
        
        def learn(self, X, y):
            
            m,n = X.shape
            min = X.min(axis=0)
            max = X.max(axis=0)
            err = sys.maxint
            
            steps = 10
            stump = {}
            for i in range(n):
                stepSize = (max[i]-min[i])/steps
                for j in range(-1,int(steps)+1):
                    for ineq in ['lt', 'gt']:
                        self.ineq = ineq
                        self.feat = i
                        self.threshold = (min[i] + float(j) * stepSize)
                        pred = self.classify(X)
                        wErr = np.average(pred != y)
                        if wErr < err:
                            err = wErr
                            stump['dim'] = self.feat
                            stump['thresh'] = self.threshold
                            stump['ineq'] = self.ineq
            
            self.ineq = stump['ineq']
            self.feat = stump['dim']
        self.threshold = stump['thresh']
        
        def classify(self, X):
            
            result = np.ones(X.shape[0])
            
            if self.ineq == 'ge':
                result[X[:,self.feat] <= self.threshold] = 0
            else:
                result[X[:,self.feat] >  self.threshold] = 0
            return np.array(result)

decision_trees = []
    
    def __init__(self, num_trees, samples=None):
        
        self.num_trees = num_trees
        self.decision_trees = [self.DecisionTree()] * num_trees
    
    def fit(self, X, y):
        
        n = np.round(np.sqrt(X.shape[1]));
        
        for i in range(self.num_trees):
            index = np.random.randint(X.shape[0], size=n)
            self.decision_trees[i].learn(X[index],y[index])

def predict(self, X):
    
    X = np.array(X)
        
        votes = np.zeros((X.shape[0],self.num_trees))
        for i in range(self.num_trees):
            votes[:,i] = self.decision_trees[i].classify(X)

    label,_ = mode(votes, axis=1)
        
        return label


def main():
    
    X = []
    y = []
    
    # Load data set
    with open("hw4-data.csv") as f:
        next(f, None)
        
        for line in csv.reader(f, delimiter = ","):
            X.append(line[:-1])
            y.append(line[-1])

X = np.array(X, dtype = float)
    y = np.array(y, dtype = int)
    
    results = []
    
    K = 10
    for k in range(K):
        X_train = np.array([x for i, x in enumerate(X) if i % K != k], dtype = float)
        y_train = np.array([z for i, z in enumerate(y) if i % K != k], dtype = int)
        X_test  = np.array([x for i, x in enumerate(X) if i % K == k], dtype = float)
        y_test  = np.array([z for i, z in enumerate(y) if i % K == k], dtype = int)
        
        randomForest = RandomForest(1)
        randomForest.fit(X_train, y_train)
        y_predicted = randomForest.predict(X_test)
        results.extend([prediction == truth for prediction, truth in zip(y_predicted, y_test)])

    # Accuracy
    accuracy = float(results.count(True)) / float(len(results))
    print "accuracy: %.4f" % accuracy

generateSubmissionFile(myname, randomForest)


main()
