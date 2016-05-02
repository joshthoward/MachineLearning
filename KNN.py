import sys, operator, heapq, numpy as np, bottleneck as bn

from scipy.spatial.distance import cdist

# Import and format data
def format(D, type):
    
    if (type == 'Office'):
        return D[:,:D.shape[1]-3], D[:, D.shape[1]-1]
    else:
        D = D.T
        return D[:,:D.shape[1]-1], D[:, D.shape[1]-1]

def readData(file):
    
    if (file == 'Wine'):
        D = np.genfromtxt('../Wine/wine.data', delimiter=',')
        D = D[np.random.randint(len(D),size=len(D))]
        m = D.shape[0]
        p = round(.8*m)
        return D[:p,1:], D[p:,1:], D[:p,0], D[p:,0]
    
    elif (file == 'Office' or file == 'MNIST'):
        train = np.genfromtxt('../' + file + '/train.csv', delimiter=',')
        test = np.genfromtxt('../' + file + '/test.csv', delimiter=',')
        trainDat, trainLab = format(train, file)
        testDat, testLab = format(test, file)
        return trainDat, testDat, trainLab, testLab

    else:
        print 'Not a valid data set.'
        sys.exit()

# Performs Data Whitening
def whiten(D):
    C = np.cov(D.T)
    V, E = np.linalg.eigh(C)
    for i in range(V.shape[0]):
        if V[i] != 0:
           V[i]  = 1/np.sqrt(np.abs(V[i]))

    return np.diag(V).dot(E.T)

# K-Nearest Neighbors implementation
class KNN:

    def __init__(self, k, trainDat, trainLab, M=None):
        
        self.k = k
        self.X = trainDat
        self.y = trainLab
        if M is None:
            self.M = np.eye(self.X.shape[1])
        else:
            self.M = M

    def predict(self, testDat):

        pred = np.zeros(testDat.shape[0])
        
        if testDat.ndim == 1:
        
            distances = cdist(testDat[np.newaxis,:], self.X, 'euclidean').ravel()
            
            index = bn.argpartsort(distances, n=self.k)
            label = self.y[index[:self.k]]
            votes = {}
            
            for i in label:
                if i in votes:
                    votes[i] += 1
                else:
                    votes[i]  = 1
        
            return max(votes.iteritems(), key=operator.itemgetter(1))[0]

        else:
            
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

# Main method to perform tests
def main(input):

    try:
        f = input[0]
        F = input[3]
        k = int(input[1])
        d = int(input[2])
    
    except IndexError:
        print "Wrong number of inputs."
        sys.exit()
    
    except TypeError:
        print "Wrong input type."
        sys.exit()

    if d == 100:
        iters = 1
    elif (d == 20 or d == 50 or d == 80):
        iters = 5
    else:
        sys.exit()

    trainDat, testDat, trainLab, testLab = readData(f)

    for i in range(iters):
         
        index = np.random.randint(trainDat.shape[0],size=round(d*trainDat.shape[0]/100))
        trainDat = trainDat[index]
        trainLab = trainLab[index]
        
        if F == 'LOO':
            
            W = whiten(trainDat)
            trainDat = trainDat.dot(W.T)

            acc = []
            for i in range(trainDat.shape[0]):
                tempTrainDat = np.delete(trainDat, i, axis=0)
                tempTrainLab = np.delete(trainLab, i, axis=0)
                
                kNearestNeighbors = KNN(k, tempTrainDat, tempTrainLab)
                predLab = kNearestNeighbors.predict(trainDat[i])
                acc.append(trainLab[i] == predLab)
    
            print 'Average leave-one-out accuracy: ', np.average(np.array(acc))
    
        elif (int(F) == 2 or int(F) == 5):

            p = round(trainDat.shape[0]/int(input[3]))
            
            W = whiten(trainDat[p:])
            tempTrainDat = trainDat[p:].dot(W.T)
            tempTrainLab = trainLab[p:]
            tempTestDat  = trainDat[:p].dot(W.T)
            tempTestLab  = trainLab[:p]

            kNearestNeighbors = KNN(k, tempTrainDat, tempTrainLab)
            predLab = kNearestNeighbors.predict(tempTestDat)
            print np.average(tempTestLab == predLab) 

        else:
            print 'Invalid cross validation scheme'
            sys.exit()

        predLab = kNearestNeighbors.predict(testDat.dot(W.T))
        print np.average(testLab == predLab)

if __name__ == '__main__':
    main(sys.argv[1:])
