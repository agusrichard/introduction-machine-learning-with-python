# -*- coding: utf-8 -*-

"""
Created on Thu Jun 13 06:14:56 2019

@author: Agus Richard Lubis
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# Two function to obtain uncertainty estimates 
# decision_function and predict_proba

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_circles
X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# the classes 'blue' and 'red'
y_named = np.array(['blue', 'red'])[y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)

# Build the gradient boosting model
gbrt = GradientBoostingClassifier(random_state=0).fit(X_train, y_train)

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(gbrt.decision_function(X_test).shape))
df = gbrt.decision_function(X_test)
print("Thresholded decision function:\n{}".format(gbrt.decision_function(X_test) > 0))
greater_zero = (gbrt.decision_function(X_test) > 0).astype(int)

# Predicting probabilities
print("Shape of probabilities: {}".format(gbrt.predict_proba(X_test).shape))
pp = gbrt.predict_proba(X_test)
print(pp[0,0] + pp[0,1])

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(13, 5)) 
mglearn.tools.plot_2d_separator(gbrt, X, ax=axes[0], alpha=.4,                                
                                fill=True, cm=mglearn.cm2) 
scores_image = mglearn.tools.plot_2d_scores(gbrt, X, ax=axes[1],                                            
                                            alpha=.4, cm=mglearn.ReBl)
for ax in axes:    
    # plot training and test points    
    mglearn.discrete_scatter(X_test[:, 0], X_test[:, 1], y_test,                             
                             markers='^', ax=ax)    
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train,                             
                             markers='o', ax=ax)    
    ax.set_xlabel("Feature 0")    
    ax.set_ylabel("Feature 1") 
cbar = plt.colorbar(scores_image, ax=axes.tolist()) 
axes[0].legend(["Test class 0", "Test class 1", "Train class 0",                
    "Train class 1"], ncol=4, loc=(.1, 1.1))

# Uncertainty in Multiclass Classification
from sklearn.datasets import load_iris

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=0)

gbrt = GradientBoostingClassifier(learning_rate=0.01, random_state=0).fit(X_train, y_train)
y_pred = gbrt.predict(X_test)
print("Score for iris dataset (Test set), modeling with GradientBoostingClassifier: {}".format(gbrt.score(X_test, y_test)))
print("Score for iris dataset (Training set), modeling with GradientBoostingClassifier: {}".format(gbrt.score(X_train, y_train)))

df = gbrt.decision_function(X_test)
pp = gbrt.predict_proba(X_test)