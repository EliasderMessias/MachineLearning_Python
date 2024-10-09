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
        #initializing parameters
        self.coef = np.zeros(X.shape[1])
        #Calculate weights and bias through QR-factorization
        Q,R = QRFactorization(X)
        #weights based on the closed form solution for OLS
        print(Q,"Q Matrix \n",R ,"R Matrix")
        self.coef = sp.solve_triangular(R,np.dot(Q.T,y))
        self.intercept = y.mean() - np.dot(self.coef,X.mean(0))

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept

class Ridge:
    def __init__(self):
        pass
