import numpy as np

from sklearn import preprocessing

# Map data set to numeric data
O = preprocessing.LabelEncoder()
H = preprocessing.LabelEncoder()
O.fit(['b','g','y','r'])
H.fit(['1:1','1:2','1:3','1:4','2:1','2:2','2:3','2:4',\
       '3:1','3:2','3:3','3:4','4:1','4:2','4:3','4:4'])

# Read data set and calculate statistics
def readData():
    
    A = np.zeros((16,16))
    B = np.zeros((16,4))
    I = np.zeros((16))
    
    #Get observation counts
    with open('../Robot/robot_train.data', 'r') as file:
        first = 0
        for line in file:
            
            if (line != '.\n'):
                first += 1
                
                split = line.strip('\n').split(" ")
                i  =  H.transform(split[0])
                B[i,O.transform(split[1])] += 1
                
                if (first != 1):
                    A[i,j] += 1
                j = i
        
            else:
                I[H.transform(split[0])] += 1
                first = 0

    # Normalize counts
    I /= np.sum(I)
    
    for i in range(A.shape[1]):
        sum = np.sum(A[:,i])
        if (sum != 0):
            A[:,i] /= sum

    for i in range(B.shape[0]):
        sum = np.sum(B[i,:])
        if (sum != 0):
            B[i,:] /= sum

    return I, A, B


def Viterbi(y, I, A, B):
    
        m = A.shape[0]
        n = y.shape[0]
    
        P = np.zeros((m,n))
        S = np.zeros((m,n))
        x = np.zeros((n))
    
        P[:,0] = I * B[:,y[0]]
        
        for i in range(1,n):
            P[:,i] = np.max(P[:,i-1, None].dot(B[:,y[i],None].T)*A, axis=0)
            S[:,i] = np.argmax(np.tile(P[:,i-1, None],m)*A, axis=0)

        x = [P[:,-1].argmax()]
        for i in range(n-1, 0, -1):
            x.append(S[x[-1],i])

        return np.array(x[::-1])

def main():

    I, A, B = readData()
    acc = []
    with open('../Robot/robot_test.data', 'r') as file:
        sequence = []
        labels   = []
        for line in file:
            if (line != '.\n'):
                temp = line.strip('\n').split(" ")
                sequence.append(temp[1])
                labels.append(temp[0])
            else:
                sequence = O.transform(sequence)
                labels = H.transform(labels)
                predicted = Viterbi(sequence,I,A,B)
                acc.append(int(np.average(labels == predicted) * 100) / 100.0)
                sequence = []
                labels = []

    print np.average(np.array(acc))


if __name__ == '__main__':
    main()
