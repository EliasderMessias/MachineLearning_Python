from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.linear import LinRegression
from src.knn import KNNRegressor

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#generate simple dataset for regression ( 1 feature only )
X, y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 1)
print(X, "X")
y = np.add(y,1000)
X = np.add(X,500)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#LinearRegression
reg = LinRegression()
reg.fit(X_train, y_train)
pred_lin = reg.predict(X)
print(reg.coef,reg.intercept)

#KNN-Regression
knn = KNNRegressor(3)
knn.fit(X_train, y_train)
#sort X, so we can use it for a plot later
X_sorted = np.sort(np.ndarray.ravel(X))
pred_knn = knn.predict(X_sorted)


#plotting the sample + predicted Lines
cmap = plt.get_cmap('viridis')
fit = plt.figure(figsize = (8,6))
m1 = plt.scatter(X_train, y_train, color = cmap(0.9), s = 10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s = 10)
plt.plot(X, pred_lin, color = 'black', linewidth = 1, label = 'Lin_Regression')
plt.plot(X_sorted, pred_knn, color = 'green', linewidth = 1, label = 'Knn_Regression')

plt.show()
