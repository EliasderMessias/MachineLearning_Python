import numpy as np
import scipy.linalg as sp

#QR decomposition using modified Gram-Schmidt, for less numerical instability than classical GS
def QRFactorization(X):
    m,n = X.shape
    R = np.empty([n,n])
    Q = np.empty([m,n])
    
    for i in range(n):
        v = X[:,i]
        for j in range(i):
            R[i,j] = np.dot(Q[:,j].T,v)
            v = v - (R[i,j]*Q[:,j])
        R[i,i] = np.linalg.norm(v)
        Q[:,i] = v/R[i,i]
    return Q,R.T

class LinRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        #We first add a row of 1s to X so we can get the intercept with the y-axis.
        Xextended = np.append(np.ones([X.shape[0],1]),X,1)

        #Calculate weights and bias through QR-factorization
        Q,R = QRFactorization(Xextended)
        #weights based on the closed form solution for OLS
        params = sp.solve_triangular(R,np.dot(Q.T,y))
        self.coef = params[1:]
        self.intercept = params[0]

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept

class Ridge:
    def __init__(self):
        pass
