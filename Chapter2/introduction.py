# -*- coding: utf-8 -*-

# Chapter 2: Supervised Machine Learning

# Classification and Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

#------------------------------------------------------------------------------
# Introduction


# Classification Problem
# Generating dataset
X, y = mglearn.datasets.make_forge() 
# Plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
print("X shape: {}".format(X.shape))


# Regression Problem
# Generating dataset
X, y = mglearn.datasets.make_wave(n_samples=40)
# Plot dataset
plt.plot(X, y, 'o')
plt.ylim(-3, 3) # Determine from what to what number
plt.xlabel("Feature")
plt.ylabel("Target")

# Breast cancer dataset (Classification Problem)
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("Cancer.keys(): \n{}".format(cancer.keys()))
print(cancer.DESCR)
print("Shape of cancer data: {}".format(cancer.data.shape))
print("Sample counts per class:\n {}".format(
        {n: v for n, v in zip(cancer.target_names, np.bincount(
                cancer.target))}))
print("Feature names:\n{}".format(cancer.feature_names))

# The Boston Housing dataset (Regression Problem)
from sklearn.datasets import load_boston
boston = load_boston()
print("Data shape: {}".format(boston.data.shape))
X, y = mglearn.datasets.load_extended_boston()
print("X.shape: {}".format(X.shape))