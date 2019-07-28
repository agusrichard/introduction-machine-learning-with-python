# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# Linear models for regression
mglearn.plots.plot_linear_regression_wave()

# Linear Regression (ordinary least squares)
from sklearn.linear_model import LinearRegression
X, y = mglearn.datasets.make_wave(n_samples=60)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
lr = LinearRegression().fit(X_train, y_train)

# Making prediction
y_pred = lr.predict(X_test)
print(lr.score(X_test, y_test))

# coefficient and intercept
print(lr.coef_)
print(lr.intercept_)

# Boston Housing Dataset
X, y = mglearn.datasets.load_extended_boston()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
lr = LinearRegression().fit(X_train, y_train)

# Ridge Regression
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
print("Test set score: {}".format(ridge.score(X_test, y_test)))
print("Training set score: {}".format(ridge.score(X_train, y_train)))

for i in range(1, 11):
    ridge = Ridge(alpha=i).fit(X_train, y_train)
    print("Test set score for alpha={}: {}".format(i, ridge.score(X_test, y_test)))
    print("Training set score for alpha={}: {}".format(i, ridge.score(X_train, y_train)))
    
# Some visualising
mglearn.plots.plot_ridge_n_samples()


#  Lasso
from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test))) 
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
# We increase the default setting of max_iter
lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso001.score(X_train, y_train))) 
print("Test set score: {:.2f}".format(lasso001.score(X_test, y_test))) 
print("Number of features used: {}".format(np.sum(lasso001.coef_ != 0)))

