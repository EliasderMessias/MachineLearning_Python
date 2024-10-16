import numpy as np

def euclid_dist(x: list, y: list) -> float:
    return np.sqrt(np.sum((x - y) ** 2))

#QR decomposition using modified Gram-Schmidt, for less numerical instability than classical GS
def qrfactorization(X: np.array) -> tuple[np.array,np.array]:
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

def normalize_data(X):
    pass