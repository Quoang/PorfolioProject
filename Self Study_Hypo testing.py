#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 11:03:35 2021

@author: poniz
"""

import pandas as pd
import numpy as np
import scipy.stats as stats

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
titanic=titanic_data
print (titanic_data.head())

#Information on the variables
print(type(titanic_data))
print(titanic_data.info())


#1) Test if the average age is equal to 30 and conclude about the test
#One sample mean Test (two sided test)
print(titanic_data.Age.mean())
print(stats.ttest_1samp(titanic_data.Age, 30, nan_policy="omit"))
#accept the null

#2) Test if the average age is equal to 35 and conclude about the test
print(stats.ttest_1samp(titanic_data.Age, 35, nan_policy="omit"))
#reject the null

#3) Test if the average Fare is equal to 30 and conclude about the test
print(titanic_data.Fare.mean())
print(stats.ttest_1samp(titanic_data.Fare, 30, nan_policy="omit"))
#accept the null

#4) Test if the average Fare is equal to 25 and conclude about the test
print(stats.ttest_1samp(titanic_data.Fare, 25, nan_policy="omit"))
#reject the null


#5) Test whether the average Fare is equal to the average age and conclude.
#two sample mean test (two sided)
print(stats.ttest_ind(titanic_data.Fare, titanic_data.Age, nan_policy="omit"))
#Conclusion: we accept the null hypothesis
#Before concluding seriously, we should check the underlying hypthesis of equal variance in the two samples
print(titanic_data.Age.std(), titanic_data.Fare.std())
#this hypiothesis is too strong, we have to run the Welch's t-test for unequal variances/ sample sizes
print(stats.ttest_ind(titanic_data.Age, titanic_data.Fare, nan_policy="omit", equal_var=False))
#we still accept the null of equal means under heterogeneous variance

#6) Test whether the average age of female was equal to the average of male. Conclude
titanic_data["F_Age"]=np.where(titanic_data["Sex"]=="female",titanic_data["Age"],float("Nan"))
titanic_data["M_Age"]=np.where(titanic_data["Sex"]=="male",titanic_data["Age"],float("Nan"))

print(stats.ttest_ind(titanic_data.F_Age, titanic_data.M_Age, nan_policy="omit"))
#Conclusion: we reject the null hypothesis
print(titanic_data.F_Age.mean(), titanic_data.M_Age.mean())
#Before concluding seriously, we should check the underlying hypthesis of equal variance in the two samples
print(titanic_data.F_Age.std(), titanic_data.M_Age.std())
#Variance are very similar
print(stats.ttest_ind(titanic_data["F_Age"], titanic_data["M_Age"], nan_policy="omit", equal_var=False))





