import numpy as np

class Normalizer():
    def __init__(self,X):
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)

    def transform(self, X):
        return np.divide(np.subtract(X, self.mean), self.std)
    
class MinMaxScaler():
    def __init__(self, min = 0, max = 1):
        self.min = min
        self.max = max

    def transform(self, X):
        return np.divide(np.subtract(X, self.max), self.max - self.min)
    
