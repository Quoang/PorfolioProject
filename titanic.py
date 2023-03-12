# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 18:43:35 2021

@author: poniz
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt 
import seaborn as sns
import statsmodels.api as sm

titanic_data = pd.read_csv("titanic.csv", header = 0, sep = ",")

print(titanic_data)

#%%
# 1) describe the quality of the information contained in the dataset 
# and generate a table of descriptive statistics

# Quantitive data: Age, Sibsb, Parch, Fare > mấy này vẽ biểu đồ được nè
# Quatitive data: Survied, Pclass, Sex, Cabin, embarked

col_names = ['Age', 'SibSp', 'Parch', 'Fare']
df = pd.DataFrame(titanic_data, columns=col_names)
descriptive_statistics=df.describe()

print(descriptive_statistics)

info = titanic_data.info()
print(info)

frequency_data = titanic_data['Pclass'].value_counts()
print(frequency_data)



#%%

#2) reduce the number of decimals to 2 for all variables
reduce_number_data = round(titanic_data,2)
print(reduce_number_data)



#%%

#3) Calculate the Skewness and Kurtosis for all continuous variables

skew = df.skew(axis = 0, skipna = True)
kurt = df.kurt(axis = 0, skipna = True)
print("Skew: " +  str(skew))
print("Kurtoness: " + str(kurt))




#%%
#4) Plot histograms for continuous variables
%matplotlib inline

sns.set(style="whitegrid")

filter_data = df.dropna(subset=['Age'])
plt.figure(figsize=(14,8))
sns.distplot(filter_data['Age'], kde=False)


filter_data = df.dropna(subset=['Fare'])
plt.figure(figsize=(14,8))
sns.distplot(filter_data['Fare'], kde=False)




#%%
#5) Generate a bar chart for the gender variable

gender_counts = titanic_data["Sex"].value_counts()
print(gender_counts)

sns.set(style='darkgrid')
plt.figure(figsize=(30,30))
ax = sns.countplot(x='Sex', data = titanic_data)



#%%
#6) Generate a bar chart and a pie chart for the Pclass variable


pclass_counts = titanic_data['Pclass'].value_counts()
print(pclass_counts)

df_pclass = pd.DataFrame({'Class_type' : pclass_counts}, 
                     index = ['class 1', 'class 2' , ' 3'])

df_pclass.plot.pie(y='Class_type', figsize=(10,10), autopct='%1.1f%%')

sns.set(style='darkgrid')
plt.figure(figsize=(30,30))
ax = sns.countplot(x='Pclass', data=titanic_data)





