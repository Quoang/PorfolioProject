#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 19:09:12 2021

@author: poniz
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot
import statsmodels.stats.diagnostic  as d
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels import IV2SLS, IVGMM
from collections import OrderedDict 
from linearmodels.iv.results import compare
from sklearn.metrics import confusion_matrix
import seaborn as sns


airfare = pd.read_csv("airfare.csv", header = 0, sep = ",")

#type: panel

print (airfare.head())

#Information on the variables
print(type(airfare))
print(airfare.info())

airfare_new=pd.get_dummies(airfare, columns= ["id"], prefix = ["lign"])
print(airfare_new.info())
print(type(airfare_new))


X1=airfare_new.iloc[:,15:-1]
X1 = sm.add_constant(X1)
Y1=airfare_new[["fare"]]
model1 = sm.OLS(Y1, X1)
results1 = model.fit()
print(results1.summary())
