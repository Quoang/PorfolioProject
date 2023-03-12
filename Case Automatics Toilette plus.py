#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:57:51 2021

@author: poniz
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.api import abline_plot
import statsmodels.stats.diagnostic  as d
from statsmodels.stats.outliers_influence import variance_inflation_factor
from linearmodels import IV2SLS, IVGMM
from collections import OrderedDict 
from linearmodels.iv.results import compare
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph
from sklearn.metrics import confusion_matrix
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn import metrics


flush_data= pd.read_excel("flush.xlsx", header=0)
flush=flush_data
print (flush.head())

# Let's check the dimensions of the dataframe
print(flush.shape)

# Let's see the type of each column
print(flush.info())

#Information on the variables
print(flush["Case of flush"].value_counts(normalize=True).plot(kind="pie", autopct='%1.1f%%', title="Case of flush distribution"))
print(flush["Case of flush"].value_counts(normalize=True).plot(kind="bar", title="Case of flush distribution"))

# this proves li dataset balanced  so its good , donc yit3alim ifarik binet categories !

#Data Pre-Processing
flush.shape
flush_des = flush.describe()
flush.dtypes
corr=flush.corr().round(2)
sns.heatmap(flush.corr(),annot=True,lw=1)

#function to standardize data
X=flush.iloc[:,4:16]
y=flush["Case_of_flush"]

# Adding up the missing values (column-wise)
flush.isnull().sum()

#Slip the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

scaler = StandardScaler()
X_train[['Blue_1_1','Blue_1_2','Red_1_1','Red_1_2','Green_1_1','Green_1_2']]=scaler.fit_transform(X_train[['Blue_1_1','Blue_1_2','Red_1_1','Red_1_2','Green_1_1','Green_1_2']])
X_train.head()

# Let's see the correlation matrix 
plt.figure(figsize = (20,10))        # Size of the figure
sns.heatmap(X_train.corr(),annot = True)
plt.show()

# Logistic regression model
logm1 = sm.GLM(y_train,(sm.add_constant(X_train)), family = sm.families.Binomial())
logm1.fit().summary()

y_train_pred = logm1.predict(X_train)
y_train_pred[:10]


#Start the PCA algorithm

#PCA






#correlation between PC's and Variables

feature_names = list(flush.columns.values)
del feature_names[0:4]
plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2,3],['1st Comp','2nd Comp','3rd Comp','4rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(feature_names)),feature_names,rotation=65,ha='left')
plt.show()

#Variance analysis ==> choosing number of PC
pca = PCA().fit(flush_new)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

#Feature vector (with all PCs)
X_pca=pca.transform(flush_new)


#Maximul Likelihood (logit model)
logistic_model = sm.GLM(flush["Case_of_flush"], X_pca, family=sm.families.Binomial())
results = logistic_model.fit()
print(results.summary())
yplogit=results.predict(X_pca)
finalX.loc[:,"yplogit"]=yplogit

logreg_acc = metrics.accuracy_score(Y,yplogit.round(0))
print(logreg_acc)

#Generate the confusion matrix
Z=finalX["yplogit"]
Y=finalX["Case_of_flush"]
c2 = metrics.confusion_matrix(Y,Z)
print(c2)

#Maximul Likelihood (logit model)
logistic_model = sm.GLM(flush["Case_of_flush"], X, family=sm.families.Binomial())
results_1= logistic_model.fit()
print(results_1.summary())
yplogit1=results_1.predict(X)
finalX.loc[:,"yplogit1"]=yplogit1

logreg_acc1 = metrics.accuracy_score(Y,yplogit1.round(0)).round(4)
print(logreg_acc1)


#Generating odds ratios
print(np.exp(results.params))


sns.heatmap(c2, annot=True,  fmt='', cmap='Blues')

#Multinomial
Mlogit = smf.mnlogit("Case_of_flush ~ Feature_vector_1 + Feature_vector_2 + Feature_vector_3", finalX).fit()
Mlogit.summary()
ymlogit=Mlogit.predict(finalX[['Feature_vector_1', 'Feature_vector_2', 'Feature_vector_3']])
print(ymlogit)
flush_2= pd.concat([finalX, ymlogit], axis=1, join="inner")
flush_2["Pred2"] =flush_2[[0,1,2]].idxmax(axis=1)
flush_2["Pred"] = flush_2["Pred2"]
Z2=flush_2["Pred"]
c4 = confusion_matrix(Y,Z2)
print(c4)

#Poisson model with PCA

poisson_model =sm.GLM(y,X_pca,family=sm.families.Poisson())
results1 = poisson_model.fit()
print(results1.summary())
ypois=results1.predict(X_pca).round(0)
print(ypois)
finalX.loc[:,"ypois"]=ypois
#Generate the confusion matrix
Z=finalX["ypois"]
Y=finalX["Case_of_flush"]
c4 = metrics.confusion_matrix(y,Z)
print(c4)
poireg_acc = metrics.accuracy_score(ypois,y)
print(poireg_acc)

#poisson model without PCA
poisson_model =sm.GLM(Y,X,family=sm.families.Poisson())
results2 = poisson_model.fit()
print(results2.summary())
ypois1=results2.predict(X).round(0)
print(ypois1)
finalX.loc[:,"ypois1"]=ypois1



K1=finalX["ypois"]
K2=finalX["Case_of_flush"]


sns.heatmap(c4/np.sum(c4), annot=True,  fmt='.2%', cmap='Blues')
sns.heatmap(c2/np.sum(c2), annot=True,  fmt='.2%', cmap='Blues')


pd.crosstab(K2, K1, rownames=['True'], colnames=['Predicted'], margins=True)

#Without PCA

#Poisson model

poisson_model =sm.GLM(Y,X,family=sm.families.Poisson())
results1 = poisson_model.fit()
print(results1.summary())
ypois=results1.predict(X_pca).round(0)
print(ypois)
finalX.loc[:,"ypois"]=ypois



#EDA
sns.pairplot(flush,hue='Case_of_flush')
sns.pairplot(flush,hue='Flush_volume')
sns.boxplot(y='Flush_volume',x='Blue_2_2',data=flush)

#dummy variables
flush_new=pd.get_dummies(flush, columns= ["Case of flush"], prefix = ["lign"])


#Multiple linear model 1
OLSX=flush[["Blue 1_1", "Blue 1_2", "Green 1_1", "Green 1_2", "Red 1_1", "Red 1_2","Blue 2_1", "Blue 2_2", "Green 2_1", "Green 2_2", "Red 2_1", "Red 2_2"]]
OLSX = sm.add_constant(OLSX)
OLSY=flush[["Case of flush"]]
ols = sm.OLS(OLSY, OLSX)
results = ols.fit()
print(results.summary())

#Multiple linear model 2
OLSX1=flush[["Blue 2_1", "Blue 2_2", "Green 2_1"]]
OLSX1 = sm.add_constant(OLSX1)
OLSY1=flush[["Case_of_flush"]]
ols1 = sm.OLS(OLSY1, OLSX1)
results1 = ols1.fit()
print(results1.summary())

##Lasso model
# define model
model = Lasso(alpha=1.0)
data = flush_data.values
X, y = data[:, 1:-1], data[:, -2]

# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = np.absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
# fit model
model = LassoCV(alphas=arange(0, 1, 0.01), cv=cv, n_jobs=-1)
model.fit(X, y)
# summarize chosen configuration
print('alpha: %f' % model.alpha_)


#poisson model

formula = "Flush_volume ~ Blue_1_1 + Blue_1_2 + Green_1_1 +  Green_1_2 + Red_1_1 + Red_1_2+ Blue_2_1 + Blue_2_2 + Green_2_1 + Green_2_2 + Red_2_1 + Red_2_2"
response, predictors = dmatrices(formula,flush,return_type="dataframe")
po_results = sm.GLM(response, predictors, family=sm.families.Poisson()).fit()
print(po_results.summary())

##logistic regression
cols=["Blue_1_1", "Blue_1_2", "Green_1_1", "Green_1_2", "Red_1_1", "Red_1_2","Blue_2_1", "Blue_2_2", "Green_2_1", "Green_2_2", "Red_2_1", "Red_2_2"] 
LX=flush[cols]
LY=flush[["Case_of_flush"]]
logit_model=sm.Logit(LY,LX)
L_result=logit_model.fit()
print(L_result.summary2())





