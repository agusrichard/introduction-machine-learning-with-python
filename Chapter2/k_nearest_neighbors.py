# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn


# -----------------------------------------------------------------------------
# k-Nearest Neighbors
# make_forge problem

mglearn.plots.plot_knn_classification(n_neighbors=5)
from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()

# Splitting dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Instantiate KNeighborsClassifier class and fitted training set 
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))

# Visualising the decision boundaries
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):    
    # the fit method returns the object self, so we can instantiate    
    # and fit in one line    
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)    
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)    
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)    
    ax.set_title("{} neighbor(s)".format(n_neighbors))    
    ax.set_xlabel("feature 0")    
    ax.set_ylabel("feature 1") 
axes[0].legend(loc=3)