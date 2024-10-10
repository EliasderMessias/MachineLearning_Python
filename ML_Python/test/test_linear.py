from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

import unittest
from src.linear import LinRegression
from sklearn import datasets
from sklearn.model_selection import train_test_split

class TestLinear(unittest.TestCase):
    def test_Initialize(self):
        self.assertEqual(LinRegression().coef, None)
        self.assertEqual(LinRegression().intercept, None)

    def test_LinRegression(self):
        #create sample and split it
        X, y = datasets.make_regression(n_samples = 100, n_features = 3, random_state = 3)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

        #setup model
        lin = LinRegression()
        lin.fit(X_train,y_train)
        y_pred = lin.predict(X_test)

        #precalculated weights and bias from sklearn LinearRegression
        real_w = [29.1931186, 75.5529479, 29.20319866]
        real_b = 0

        #test weights and bias vs sklearns method parameters
        for i in range(lin.coef.shape[0]):
            self.assertAlmostEqual(lin.coef[i], real_w[i], delta = 0.01)
        self.assertAlmostEqual(lin.intercept, real_b, delta = 0.01)

        #precalculated predictions for specific sample
        real_pred = [-133.41330752, -144.58097038, 158.63138463, 54.63284421, 67.02457604, -9.19545614, 93.41791274, 90.52886868, -144.5134964, 62.20363551, 60.93164245, -184.51696485, -42.47252346, -14.03849722, 149.02638913, 91.45080688, 100.12714128, -224.52846856, -150.39202857, 193.13524936, 156.654647, -61.51985597, 3.09583699, -106.35366146, -89.30487289]
        #test the predictions
        for i in range(len(y_test)):
           self.assertAlmostEqual(y_pred[i], real_pred[i], delta = 0.01)


if __name__ == "__main__":
    unittest.main(verbosity=2)