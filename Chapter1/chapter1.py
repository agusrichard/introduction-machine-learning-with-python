# -*- coding: utf-8 -*-

# Testing bomb site

import numpy as np

# Simpy just creating an array. The syntax 
# looks precisely like matlab
x = np.array([[1,2,3], [4,5,6], [7,8,9]])
print("x: \n{}".format(x))
x_sparse = sparse.coo_matrix(x)
print(x_sparse)

# Creating indentity matrix
from scipy import sparse
eye = np.eye(4)
print("Numpy array: \n {}".format(eye))

# Convert numpy array to a scipy matrix  in SCR format
sparse_matrix = sparse.csr_matrix(eye)

# Another application
data = np.ones(4)
row_indices = np.arange(4)
col_indices= np.arange(4)
eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
print(eye_coo)

%matplotlib inline
import matplotlib.pyplot as plt

# Generate a sequence of numbers from -10 to 10 with 100 steps in between
x = np.linspace(-10, 10, 100)
print(x)
# Create a second array using sine
y = np.sin(x)
print(y)
# The plot function
plt.plot(x, y, marker='.')


import pandas as pd
from IPython.display import display
# Create a simple dataset of people
data = {'Name': ["John", "Anna", "Peter", "Linda"],        
        'Location' : ["New York", "Paris", "Berlin", "London"],        
        'Age' : [24, 13, 53, 33]}

data_pandas = pd.DataFrame(data)
display(data_pandas)
display(data_pandas[data_pandas.Age > 30])

# Meet dataset of Iris (Such a great song isn't it?)
from sklearn.datasets import load_iris
iris_dataset = load_iris()
print(iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")
print("Target names: {}".format(iris_dataset['target_names']))
print("Feature names: \n{}".format(iris_dataset['feature_names']))
print("Type of data: {}".format(type(iris_dataset['data'])))
print("First five rows of data:\n{}".format(iris_dataset['data'][:5]))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset['data'], iris_dataset['target'], random_state=0)

# Create dataframe from data in X_train
# Label the columns using the strins
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
# Create a scatter matrix from dataframe, color by y_train
pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15, 15),
                           marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)

# Building k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

# Making predictions
X_new = np.array([[5, 2.9, 1, 0.2]])
print(X_new.shape)
prediction = knn.predict(X_new)
print("PredictionL {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset[
        'target_names'][prediction]))
y_pred = knn.predict(X_test)
print("Test set predictions: {}".format(y_pred))
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))




