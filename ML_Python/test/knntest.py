
from pathlib import Path
import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from src.knn import KNNClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state=0)

knn = KNNClassifier(3)

knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))




