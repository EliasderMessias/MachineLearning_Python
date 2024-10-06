from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import unittest
import numpy as np
from src.linear import LinRegression

class TestLinear(unittest.TestCase):
    def test_internalParam(self):
        self.assertEqual(LinRegression().lrate, 0.001)
        self.assertEqual(LinRegression(lrate = 0.01).lrate, 0.01)
        self.assertEqual(LinRegression().n_iters, 1000)
        self.assertEqual(LinRegression(n_iters = 2000).n_iters, 2000)
    
    def test_LinRegression(self):
        for j in range(1000)
            X_train[j] = [j + 1, j + 2]   #Fix to make array bigger
        y_train = np.array([2, 3, 4, 5, 6])
        X_test = np.array([[2.5, 3.5], [4.5, 5.5]])

        lin = LinRegression(lrate = 0.001)
        lin.fit(X_train,y_train)
        
        real_weights = [1, 1]
        real_bias = [-1, 0]
        real_predict = [3.5, 5.5]

        for i in range(len(lin.weights)):
            self.assertAlmostEqual(lin.predict(X_test)[i], real_predict[i], delta = 0.02)
            self.assertAlmostEqual(lin.weights[i], real_weights[i], delta = 0.02) 
            self.assertAlmostEqual(lin.bias[i], real_bias[i], delta = 0.02) 

if __name__ == "__main__":
    unittest.main(verbosity=2)