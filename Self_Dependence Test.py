#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 20:56:44 2021

@author: poniz
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")
print (titanic_data.head())

#Information on the variables
print(type(titanic_data))
print(titanic_data.info())

#1) Test if the variance of age between male and female is the same.
#Then implement a mean test comparison for the age of female and male using appropriate option. 
titanic_data["F_Age"]= np.where(titanic_data["Sex"]=="female",titanic_data["Age"], float("NaN"))
titanic_data["M_Age"]= np.where(titanic_data["Sex"]=="male",titanic_data["Age"], float("NaN"))
print(stats.ttest_ind(titanic_data["F_Age"], titanic_data["M_Age"], nan_policy="omit", equal_var=False))
print(titanic_data.F_Age.std(), titanic_data.M_Age.std())
print(stats.ttest_ind(titanic_data["F_Age"], titanic_data["M_Age"], nan_policy="omit"))

#2) JB test on age and Fare, conclude
titanic=titanic_data.dropna(subset=["Age"])
print(stats.jarque_bera(titanic.Age))
print(stats.jarque_bera(titanic_data.Fare))

#3) SW test on age and Fare, conclude
print(stats.shapiro(titanic.Age))

print(stats.shapiro(titanic_data.Fare))

#4) Generate a sample of 10,000 observations following N(10,4). Implement the JB test and conclude
x=np.random.normal(10,4,10000)
print(stats.shapiro(x))

#5) Generate a sample of 10,000 observations following an exponential distribution with parameter 4. Implement the JB test and concludex=np.random.normal(10,4,10000)
y=np.random.exponential(4,10000)
print(stats.shapiro(y))

#6) Transform the variable “Sex” into a binary variable
titanic_data["Sex_Binary"]=np.where(titanic_data["Sex"]=="female",1,0)

#7) Test the dependence between the age and the Pclass using the appropriate test. Conclude



#8) Test the dependence between the Fare and the Pclass using the appropriate test. Conclude
#9) Test the dependence between the Fare and the gender using the appropriate test. Conclude
#10) Test the dependence between the gender and the Pclass using the appropriate test. Conclude
#11) Test the dependence between the survival and the Pclass using the appropriate test. Conclude
#12) Test the dependence between the age and the Gender using the appropriate test. Conclude
#13) Test the dependence between the age and Fare using the appropriate test. Conclude
