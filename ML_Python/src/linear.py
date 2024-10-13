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

#Proximatorfunction for the Lasso problem
def soft_threshholding_operator(omega, theta):
    return np.sign(omega)* np.maximum(np.abs(omega)- theta, 0)

class LinRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        #We first add a row of 1s to X so we can get the intercept with the y-axis.
        Xextended = np.append(np.ones([X.shape[0],1]),X,1)

        #Calculate weights and bias through QR-factorization
        Q,R = QRFactorization(Xextended)
  
        #weights based on the closed form solution for OLS:
        params = sp.solve_triangular(R,np.dot(Q.T,y))
        self.coef = params[1:]
        self.intercept = params[0]

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept

class Ridge:
    def __init__(self, λ = 1):
        self.coef = None
        self.intercept = None
        self.λ = λ

    def fit(self, X, y):
        #Create Blockmatrix of the Form [[1, X], [0, L]], where L = λ*I (from Regularisation)
        Matrix = np.block(
            [
                [np.ones((X.shape[0],1)), X],
                [np.zeros((X.shape[1],1)), np.multiply(np.sqrt(self.λ), np.eye(X.shape[1]))]
            ]
        )

        #As in LinReg use QR decomposition to solve the system 
        Q,R = QRFactorization(Matrix)
        
        y = np.append(y,np.zeros((X.shape[1],1)))
        params = sp.solve_triangular(R,np.dot(Q.T,y))
        self.coef = params[1:]
        self.intercept = params[0]
    
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
    
class Lasso:
    def __init__(self, maxiter = 3000,  l = 1):
        self.maxiter = maxiter
        self.l = l

    def fit(self, X, y):
        Xextended = np.append(np.ones([X.shape[0],1]),X,1)
        self.intercept = None
        coef = np.zeros(Xextended.shape[1])
        L = np.linalg.norm(Xextended) ** 2

        for _ in range(self.maxiter):
            coef = soft_threshholding_operator(coef - np.dot(Xextended.T, np.dot(Xextended,coef) - y) / L, self.l/L)
        self.intercept = coef[0]    
        self.coef = coef[1:]

    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept

        
