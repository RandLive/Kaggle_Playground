# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 23:57:10 2017

@author: Dream
"""
# =============================================================================
# 
# =============================================================================
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
from datacleaner import autoclean

import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.model_selection import train_test_split

from tpot import TPOTRegressor


# =============================================================================
# 
# =============================================================================
train_data= pd.read_csv("../input/train.csv")
test_data = pd.read_csv("../input/test.csv")

train = autoclean(train_data)
test = autoclean(test_data)

# =============================================================================
# 
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(train.SalePrice, train.drop('SalePrice', axis=1),
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=5, population_size=20, verbosity=2,
                      config_dict='TPOT light')
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_mnist_pipeline.py')

