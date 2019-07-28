# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

# Random forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42) 




