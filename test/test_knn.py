import numpy as np
import unittest

from src.knn import KNNClassifier, KNNRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class TestKNN(unittest.TestCase):       
    def test_internalNeighbourParam(self):
        self.assertEqual(KNNClassifier(3).k, 3)
        self.assertEqual(KNNClassifier(1).k, 1)
        with self.assertRaises(ValueError):
            KNNClassifier(-1)
    
    def test_KNNClassifier(self):
        iris_dataset = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)    
        knn = KNNClassifier(3)
        knn.fit(X_train,y_train)
        #testing against scikits prediction 
        self.assertTrue((knn.predict(X_test) == [2,1,0,2,0,2,0,1,1,1,2,1,1,1,1,0,1,1,0,0,2,1,0,0,2,0,0,1,1,0,2,1,0,2,2,1,0,2]).all()) 

    def test_KNNRegressor(self):
        X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
        y_train = np.array([2, 3, 4, 5, 6])
        X_test = np.array([[2.5, 3.5], [4.5, 5.5]])

        knn = KNNRegressor(2)
        knn.fit(X_train,y_train)
        self.assertTrue((knn.predict(X_test) == [3.5,5.5]).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)


