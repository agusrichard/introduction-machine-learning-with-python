# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# make_blobs function
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

from sklearn.svm import LinearSVC

linear_svm = LinearSVC().fit(X, y)

mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")





