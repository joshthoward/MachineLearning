import sys
import numpy as np

from sklearn.tree import DecisionTreeClassifier

np.set_printoptions(threshold=np.nan)

def formatMNIST(D):
    D = D.T
    n = D.shape[1]
    Lab = D[:, n-1]
    Dat = D[:,:n-1]
    
    return Dat,Lab

def formatOffice(D):
    n = D.shape[1]
    Lab = D[:, n-1]
    Dat = D[:,:n-3]
    
    return Dat,Lab

def readData(file):
    
    if (file == 'MNIST'):
        train = np.genfromtxt('../MNIST/train.csv', delimiter=',')
        test = np.genfromtxt('../MNIST/test.csv', delimiter=',')
        trainDat, trainLab = formatMNIST(train)
        testDat, testLab = formatMNIST(test)
        return trainDat, testDat, trainLab, testLab
    
    if (file == 'Office'):
        train = np.genfromtxt('../Office/train.csv', delimiter=',')
        test = np.genfromtxt('../Office/test.csv', delimiter=',')
        trainDat, trainLab = formatOffice(train)
        testDat, testLab = formatOffice(test)
        
        return trainDat, testDat, trainLab, testLab

    else: sys.exit()


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

def main(input):
    
    if (len(input) < 1):
        sys.exit()

    if (len(input) < 3):
        input.append(1)
        input.append(1)

    # Read data file
    trainDat, testDat, trainLab, testLab = readData(input[0])
    names = np.unique(np.concatenate((trainLab,testLab),axis=0))

    # Initialize Classifier
    A = AdaBoost(int(input[1]), int(input[2]), names)
    A.Train(trainDat, trainLab)
    P = A.Test(testDat)

    print np.average(P != testLab)

if __name__ == "__main__":
    main(sys.argv[1:])
