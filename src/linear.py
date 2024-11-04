import numpy as np
import scipy.linalg as sp
from  utils.utilfunctions import qrfactorization, vsoft_threshholding_operator, softmax

#----------------------------------Regression-------------------------------------------------------------------

class LinRegression:
    def __init__(self):
        self.coef = None
        self.intercept = None

    def fit(self, X, y):
        #we first add a row of 1s to X so we can get the intercept with the y-axis.
        Xextended = np.append(np.ones([X.shape[0],1]), X, 1)

        #calculate weights and bias through QR-factorization
        Q,R = qrfactorization(Xextended)
  
        #weights based on the closed form solution for OLS:
        params = sp.solve_triangular(R, np.dot(Q.T,y))
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
        #create blockmatrix of the form [[1, X], [0, L]], where L = λ*I (from regularization)
        Matrix = np.block(
            [
                [np.ones((X.shape[0],1)), X],
                [np.zeros((X.shape[1],1)), np.multiply(np.sqrt(self.λ), np.eye(X.shape[1]))]
            ]
        )

        #as in LinReg use QR decomposition to solve the system 
        Q,R = qrfactorization(Matrix)
        
        y = np.append(y, np.zeros((X.shape[1], 1)))
        params = sp.solve_triangular(R, np.dot(Q.T, y))
        self.coef = params[1:]
        self.intercept = params[0]
    
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept
    
class Lasso:
    def __init__(self, maxiter = 3000, regulizationparameter = 0.001 , learningrate = None):
        self.maxiter = maxiter
        self.l = regulizationparameter
        self.learningrate = learningrate

    #fit methods uses iterative shrinkage-threshholding algorithm (FISTA) -- special case of proximal gradient descent for lasso
    #state of the art is coordinate descent algorithm to solve the lasso problem ( implemented later )
    def fit(self, X, y):
        #extend X so we can calculate intercept during loop
        X_extended = np.column_stack([np.ones(X.shape[0]), X])
        coef = np.zeros(X_extended.shape[1])

        #if we didnt specify a learningrate we use the largest eigenvalue of X.T*X/len(X) as the standard
        if not self.learningrate:
            eig = np.linalg.eigvalsh(np.dot(X_extended.T, X_extended) / len(y))
            self.learningrate = 1 / np.max(eig)      

        #set initial params for FISTA
        theta = 1
        coef_prev = np.copy(coef)

        for _ in range(self.maxiter):
            y_pred = np.dot(X_extended, coef)
            grad = np.dot(X_extended.T, (y_pred - y)) / len(y)

            #intercept update
            coef[0] -= self.learningrate * grad[0] 
            #update feature coefficients with soft-thresholding (regularization)
            coef[1:] = vsoft_threshholding_operator(coef[1:] - self.learningrate * grad[1:], self.l * self.learningrate)               

            #following steps till end of loop are the modifications for accelerated ISTA
            theta_prev = np.copy(theta)
            theta = (1 + np.sqrt(1 + 4* theta ** 2)) / 2
       
            diff = coef - coef_prev
            coef_prev = np.copy(coef)
            
            coef = coef + (theta_prev - 1)/theta * diff
            
        self.intercept = coef[0]
        self.coef = coef[1:]
        
    def predict(self, X):
        return np.dot(X, self.coef) + self.intercept

#-------------------------Classification---------------------------------------------------------------------------------------------

#Implementation of (multinomial) LogisticRegression with l1 regularization
class LogisticRegression:
    def __init__(self, maxiter = 3000, regulizationparameter = 0.01, learningrate = 0.001):
        self.coef = None
        self.intercept = None
        self.maxiter = maxiter
        self.regularizationp = regulizationparameter
        self.learningrate = learningrate
        self.n_classes = None
    
    def one_hot_encode(self, y):
        y_one_hot = np.zeros([y.size, np.unique(y).size])
        y_one_hot[np.arange(y.size),y] = 1
        return y_one_hot
    
    def fit(self, X, y):
        #fit method using SAGA algorithm (reference: https://arxiv.org/pdf/1407.0202)
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        #initialize coef matrix and derivative table
        coef = np.zeros([n_features + 1, self.n_classes])
        derivative_table = np.zeros([n_samples, n_features + 1, self.n_classes])

        #extend feature matrix with row of ones to incorporate intercept term
        X_extended = np.column_stack([np.ones(X.shape[0]), X])

        #onehot encode y 
        y_one_hot = self.one_hot_encode(y)

        #fill derivative table through non-randomized run through all indexes
        for i in range(n_samples):
            y_pred = softmax(np.dot(X_extended[i], coef))
            grad = np.outer(X_extended[i], y_pred - y_one_hot[i])
        
            # Update the derivative table
            derivative_table[i] = grad

        # Use the cumulative average gradient as the starting average gradient for SAGA updates
        avg_grad = np.mean(derivative_table, axis = 0)
        
        #SAGA algorithm 
        for _ in range(self.maxiter):
            #choose a random index j uniformly from the sameples
            j = np.random.randint(n_samples)

            #note down the previous gradient for that index from our indextable
            prev_grad = derivative_table[j]

            #calculate the new gradient for our index j
            y_pred_j = softmax(np.dot(X_extended[j], coef))
            grad = np.outer(X_extended[j], y_pred_j - y_one_hot[j])

            #update the derivative table
            derivative_table[j] = grad

            #calculate the average gradient of the derivative table
            avg_grad += (grad - prev_grad)/n_samples

            #update our coefficients with the following rule (SAGA update)
            coef -= self.learningrate * (grad - prev_grad + avg_grad)

            #use the soft_threshholding_operator to apply l1 regularization to the coef
            coef = vsoft_threshholding_operator(coef, self.regularizationp * self.learningrate)

        self.coef = coef[1:]
        self.intercept = coef[0]
    
    def predict(self, X):
        #calculate class probabilities
        self.probabilities = softmax(np.dot(X, self.coef) + self.intercept)
        return np.argmax(self.probabilities, axis=1)





