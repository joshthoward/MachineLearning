import sys, numpy as np

from util import *

# Naive Bayes Classifier
class NaiveBayes:
    
    # Class constructor (train data)
    def __init__(self, D):
        
        C = D[:, 0]
        D = D[:,1:]
        
        m = np.unique(C).shape[0]
        n = D.shape[1]
        
        Avg = np.zeros((m,n))
        Var = np.zeros((n,))
        
        for j in range(n):
            Var[j] = np.var(D[:,j])
            for i in np.unique(C):
                Avg[i,j] = np.mean(D[C==i,j])
    
        self.Avg = Avg
        self.Var = Var
    
    # Class assignments (test data)
    def Test(self, D):
        
        C = D[:, 0]
        D = D[:,1:]
        n = D.shape[1]
        
        W = np.zeros(n,)
        W0 = 0
        for j in range(n):
            if (self.Var[j] != 0):
                W[j] = (self.Avg[0,j]-self.Avg[1,j])/self.Var[j]
                W0 += (np.power(self.Avg[1,j],2)-np.power(self.Avg[0,j],2))/(2*self.Var[j])
        
        np.seterr()
        
        Assignment = []
        for i in D:
            Ex = np.exp(W0 + W.T.dot(i))
            Pr = Ex/(1+Ex)
            if (Pr < .5):
                Assignment.append(1)
            else:
                Assignment.append(0)

        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(D.shape[0]):
            if (Assignment[i] == C[i]):
                if (C[i] == 0):
                    TP += 1
                else:
                    TN += 1
            else:
                if (C[i] == 0):
                    FP += 1
                else:
                    FN += 1
        
        print np.array([[TP,FP],[FN,TN]])
        return (TP + TN)/D.shape[0]

def main(argv):
    
    if (argv[0].upper() == 'WINE'):
        
        D = np.genfromtxt('wine.data', delimiter=',')
        D = SelectClass(D, 1, 2)
        
        if (argv[1] == '10-fold'):
            
            D = D[np.random.randint(len(D),size=len(D))]
            inc = int(np.floor(D.shape[0]/10))
            for i in range(0, D.shape[0], inc):
                index = np.arange(i,i+inc)
                train = D[index]
                test = np.delete(D,index,axis=0)
                NB = NaiveBayes(train)
                Acc = NB.Test(test)
    
        else:
            
            count = int(argv[1])
            
            C  = D[:,0]
            C0 = D[C==0]
            C1 = D[C==1]
            
            I0 = np.arange(C0.shape[0])
            I1 = np.arange(C1.shape[0])
            np.random.shuffle(I0)
            np.random.shuffle(I1)
            
            train = np.concatenate((C0[I0[:count]],C1[I1[:count]]), axis=0)
            test  = np.concatenate((C0[I0[count:]],C1[I1[count:]]), axis=0)
            
            NB = NaiveBayes(train)
            Acc = NB.Test(test)

    if (argv[0].upper() == 'MNIST'):
    
        train = np.genfromtxt('train.csv', delimiter=',')
        train = SelectClass(train.T, int(argv[1]), int(argv[2]), 'MNIST')

        test = np.genfromtxt('test.csv', delimiter=',')
        test = SelectClass(test.T, int(argv[1]), int(argv[2]), 'MNIST')
        
        NB = NaiveBayes(train)
        Acc = NB.Test(test)

if __name__ == '__main__':
    main(sys.argv[1:])