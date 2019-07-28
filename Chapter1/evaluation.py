# -*- coding: utf-8 -*-

# Testing for understanding

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset sample from sklearn
from sklearn.datasets import load_iris
iris_dataset = load_iris()
X = iris_dataset['data']
y = iris_dataset['target']

# Splitting dataset into training set and test set
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)


# Create dataframe from data in X_train
# Label the columns using the strins
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# Create a scatter matrix from dataframe, color by y_train !!! Need to look for this scatter matrix function!!!
pd.plotting.scatter_matrix(iris_dataframe, 
                           c = y_train, figsize = (15, 15),
                           marker='o', 
                           hist_kwds={'bins': 20}, 
                           s=60, 
                           alpha=.8)

# Building k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making prediction for a new result
X_new = np.array([[2.5, 3, 2.1, 1.9]])
print("The result of the predcition of [2.5, 3, 2.1, 1.9] is {}".format(knn.predict(X_new)))
print(iris_dataset.target_names[knn.predict(X_new)])

# Applying OUR model to test set
y_pred = knn.predict(X_test)
print(np.mean(y_pred == y_test))
print(knn.score(X_test, y_test))

