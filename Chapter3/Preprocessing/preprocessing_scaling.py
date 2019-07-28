# -*- coding: utf-8 -*-

"""
Created on Fri Jun 14 19:58:39 2019

@author: Agus Richard Lubis
"""

# Importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
from sklearn.model_selection import train_test_split 

# SVM and neural network are sensitive to data scaling
mglearn.plots.plot_scaling()

# StandardScaler ensures that for each feature the mean is 0 and the variance is 1

# RobustScaler uses median and quartiles. Ignores outliers

# MinMaxScaller makes all the data put inside the rectangle between 0 and 1

# Normalizer scales eachdata point such that the feature vector has a euclidean length of 1
# It projects the data on a circle o length 1
# Used the only direction (or angle) of the data matters


# Applying data transformations
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

# Data splitting 
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

print(X_train.shape)
print(X_test.shape)

# Scaling the data with MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# Transform data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Effect of preprocessing
from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

svm = SVC(C=100)
svm.fit(X_train_scaled, y_train)
print("Test set accuracy: {}".format(svm.score(X_test_scaled, y_test)))










