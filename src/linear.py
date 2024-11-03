import numpy as np
import scipy.linalg as sp
from  utils.utilfunctions import qrfactorization, vsoft_threshholding_operator, softmax

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


#Implementation of (multinomial) LogisticRegression with l1 regularization
class LogisticRegression:
    def __init__(self, maxiter = 1000, regulizationparameter = 0.01, learningrate = 0.001):
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
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        coef = np.zeros([n_features + 1, self.n_classes])

        X_extended = np.column_stack([np.ones(X.shape[0]), X])
        y_one_hot = self.one_hot_encode(y)
        y_pred = softmax(X_extended @ coef)

        grad = -np.sum(np.multiply(np.subtract(y_one_hot, y_pred), X_extended), axis = 0) 

        derivative_table = np.zeros([n_samples, n_features + 1])
        

        #SAGA algorithm 
        for _ in range(self.maxiter):
            j = np.random.randint(n_samples)

            y_pred = softmax(X_extended @ coef)

            prev_grad = derivative_table[j]
            grad = -np.sum(np.multiply(np.subtract(y_one_hot, y_pred), X_extended), axis = 0) 
            derivative_table[j] = grad
            avg_grad = np.mean(derivative_table, axis = 0)

            coef -= self.learningrate * (grad - prev_grad - avg_grad)

            coef = vsoft_threshholding_operator(coef, self.regularizationp*self.learningrate)

        self.coef = coef[1:]
        self.intercept = coef[0]
        print(coef)

    def predict(self, X):
        return np.argmax(softmax(np.dot(X, self.coef) + self.intercept), axis = 1)
    
    def fit2(self, X, y):
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))

        # Initialize coefficient matrix for each class and intercept
        coef = np.zeros([n_features + 1, self.n_classes])

        # Extend X for the intercept term
        X_extended = np.column_stack([np.ones(X.shape[0]), X])
        y_one_hot = self.one_hot_encode(y)
        
        # Initialize memory for past gradients
        derivative_table = np.zeros([n_samples, n_features + 1, self.n_classes])

        # SAGA main loop
        for _ in range(self.maxiter):
            # Randomly select one sample
            j = np.random.randint(n_samples)
            
            # Current prediction for sample j
            y_pred_j = softmax(np.dot(X_extended[j], coef))

            # Compute the gradient for the selected sample j
            grad_j = np.outer(X_extended[j], y_pred_j - y_one_hot[j])
            
            # Update the derivative table for sample j
            prev_grad_j = derivative_table[j]
            derivative_table[j] = grad_j

            # Calculate the average gradient across all samples in derivative table
            avg_grad = np.mean(derivative_table, axis=0)

            # SAGA update rule for the coefficients
            coef -= self.learningrate * (grad_j - prev_grad_j + avg_grad)

            # Apply soft thresholding for L1 regularization
            coef = vsoft_threshholding_operator(coef, self.regularizationp * self.learningrate)

        # Separate out the intercept and coefficients
        self.coef = coef[1:]
        self.intercept = coef[0]
    
    def predict2(self, X):
        # Calculate class probabilities
        probabilities = softmax(np.dot(X, self.coef) + self.intercept)
        return np.argmax(probabilities, axis=1)



from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

X, y = make_classification(n_samples = 300, n_features= 2, n_clusters_per_class=1, n_redundant=0,n_repeated=0, n_classes = 3, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

logreg = LogisticRegression()
logreg.fit2(X_train,y_train)
pred = logreg.predict2(X_test)

print(np.subtract(pred, y_test))



