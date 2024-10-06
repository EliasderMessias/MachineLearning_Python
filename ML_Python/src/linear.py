import numpy as np

class LinRegression:
    def __init__(self, lrate = 0.001, n_iters = 1000):
        self.lrate = lrate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        #initializing parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #simple gradient descent algo for minimizing the function y - y_pred
        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias

            self.weights = self.weights - self.lrate * 2/n_samples * np.dot((y_pred - y).T, X)
            self.bias = self.bias - self.lrate * 2/n_samples * np.sum(y_pred - y)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


