import numpy as np
from collections import Counter
from statistics import mode

def euclid_dist(x,y):
    return np.sqrt(np.sum((x-y)**2))

    

class _KNN:   #class that mirrors functionality of scikitlearns ML methods for k-Nearest-Neighbours
    def __init__(self, k = 3):
        if k > 0:
            self.k = k
        elif k <= 0:
            raise ValueError("Neighbourparameter cannot be <= 0")
        else:
            pass

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y


class KNNClassifier(_KNN):

    def predict(self, X):
        prediction = [self._predict(x) for x in X]
        return np.array(prediction)
    
    def _predict(self, x):
        #calculate distance to all points in X_train
        distances = [euclid_dist(x,x_train) for x_train in self.X_train]   

        #sort for shortest distance and take array of the k-first
        indices = np.argsort(distances)[:self.k]
        
        #give the labels for classification from the indices of the previous step
        labels = [self.y_train[i] for i in indices]

        #make a decision by majority of occuring labels
        predicted_label = mode(labels)
        return predicted_label

        
        


   


