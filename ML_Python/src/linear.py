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
    return np.sign(omega)*np.maximum(np.abs(omega)-theta,0)

#vectorize function for elementwise usage
vsoft_threshholding_operator = np.vectorize(soft_threshholding_operator)

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
    def __init__(self, maxiter = 3000, l = 0.001 , learningrate = None):
        self.maxiter = maxiter
        self.l = l
        self.learningrate = learningrate

    #fit methods uses iterative shrinkage-threshholding algorithm (FISTA) -- special case of proximal gradient descent for lasso
    #State of the art is coordinate descent algorithm to solve the lasso problem ( implemented later )
    def fit(self, X, y):
        X_extended = np.column_stack([np.ones(X.shape[0]), X])
        coef = np.zeros(X_extended.shape[1])

        #if we didnt specify a learningrate we use the largest eigenvalue of X.T*X/len(X) as the standard
        if not self.learningrate:
            eig = np.linalg.eigvalsh(np.dot(X_extended.T,X_extended)/len(y))
            self.learningrate = 1/np.max(eig)      

        #set initial params for FISTA
        theta = 1
        coef_prev = coef

        for _ in range(self.maxiter):
            y_pred = np.dot(X_extended, coef)
            grad = np.dot(X_extended.T, (y_pred - y)) / len(y)

            #intercept update
            coef[0] -= self.learningrate * grad[0] 
            #update feature coefficients with soft-thresholding (regularization)
            coef[1:] = vsoft_threshholding_operator(coef[1:] - self.learningrate * grad[1:], self.l * self.learningrate)
            

            #following steps till end of loop are the modifications for accelerated ISTA
            theta_prev = theta
            theta = (1 + np.sqrt(1 + 4* theta ** 2))/2
       
            diff = coef - coef_prev
            coef_prev = coef
            
            coef = coef + (theta_prev - 1)/theta * diff
            
        self.intercept = coef[0]
        self.coef = coef[1:]
        
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
