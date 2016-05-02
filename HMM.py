import numpy as np

'''
Viterbi Decoding of Hidden Markov Model (HMM) with parameters 
I - Initial probabilities
A - Transition probabilities
B - Emission probabilities
y - Sequence of observations
'''

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
