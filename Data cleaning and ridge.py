#!/usr/bin/env python
# coding: utf-8

# In[108]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import kurtosis, skew
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler 


# In[109]:


group_work = pd.read_excel("File-8-Group8-1.xlsx", header = 0)
group_work.head()


# # Do we have to clean our data? let's see

# In[110]:


group_work.drop(group_work.columns[[0,1,2,3,16,17,18,19]], axis=1).plot.hist(200,50);


# In[111]:


print(group_work["Case of flush"].value_counts(normalize=True).plot(kind="pie", autopct='%1.1f%%', title="Case of flush distribution")) 


# In[112]:


matrix_group_work =group_work.drop(group_work.columns[[0, 1, 2,3,16,17,19]], axis = 1)
corrMatrix = matrix_group_work.corr() 
# I want only to have case of flush as a row ! so I did the code below
corrMatrix2=corrMatrix.drop(corrMatrix.columns[[0,1,2,3,4,5,6,7,8,9,10,11]], axis= 0)
corrMatrix2


# # Clean the data

# the dataset is balanced, but case of flush = 0 represents less than 1% of the total data, so we are going to try to remove it and see if we get better results

# In[113]:


indexNames = group_work[group_work['Case of flush'] == 0 ].index
# Delete these row indexes from dataFrame
clean_data2=group_work.drop(indexNames)


# In[114]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X=  group_work.drop(group_work.columns[[0,1,2,3,16,17,18,19]], axis=1)
y=group_work['Case of flush']
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                              test_size=0.20)
reg = LinearRegression().fit(X, y)
reg.score(X, y)


# In[115]:


reg.score(X_train,y_train)


# In[116]:



from sklearn.svm import  LinearSVC
# setup random seed
np.random.seed(42)
#make the data
X1=  group_work.drop(group_work.columns[[0,1,2,3,16,17,18,19]], axis=1)
y1= group_work['Case of flush']
# split data
X1_train , X1_test ,y1_train, y1_test = train_test_split(X1,y1,test_size=0.25)
#initiate LienarSVC
clf = LinearSVC()
clf.fit(X1_train, y1_train)
# Evaluate LinearSVC
clf.score (X_test, y_test)


# In[117]:


# let's try the ridge regression model 
from sklearn.linear_model import Ridge
# Setup a random seed
np.random.seed(42)

# create the Data
X3= group_work.drop(group_work.columns[[0,1,2,3,16,17,18,19]], axis=1)
y3 = group_work['Case of flush']


# split into train and test set
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size = 0.275)
sc=StandardScaler()
X3_train= sc.fit_transform(X3_train)
X3_test= sc.transform(X3_test)
#Instantiate Ridge model
model3 = Ridge()
model3.fit(X3_train, y3_train)
# check the score of the ridge model on test data
model3.score(X3_test, y3_test)


# In[118]:


model3.score(X3_train,y3_train)


# In[119]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , mean_squared_error, r2_score
ridge_predict=model3.predict(X3_test)
y_ridge = ridge_predict+0.5
RMSE= np.sqrt(mean_squared_error(y3_test,ridge_predict))
ridge_r2=r2_score(y3_test,ridge_predict)
ridge_accuracy=accuracy_score(ridge_predict.round(0),y3_test)
accu= accuracy_score(ri)
print(RMSE)
print(ridge_r2)
print(ridge_accuracy)


# In[120]:


y3_preds=model3.predict(X3_test)
y3_preds


# In[121]:


print(len(y3_test))
print(len(y3_preds))


# # Let's try ridge with dataset that does not contain case of flush = 0 and see the difference

# In[122]:


# let's try the ridge regression model 
from sklearn.linear_model import Ridge
# Setup a random seed
np.random.seed(42)

# create the Data
X4= clean_data2.drop(clean_data2.columns[[0,1,2,3,16,17,18,19]], axis=1)
y4 = clean_data2['Case of flush']


# split into train and test set
X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size = 0.2)
X4_train= sc.fit_transform(X4_train)
X4_test= sc.transform(X4_test)
#Instantiate Ridge model
model4 = Ridge()
model4.fit(X4_train, y4_train)
# check the score of the ridge model on test data
model4.score(X4_test, y4_test)


# In[123]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , mean_squared_error, r2_score
ridge_predict=model4.predict(X4_test)
RMSE= np.sqrt(mean_squared_error(y4_test,ridge_predict))
ridge_r2=r2_score(y4_test,ridge_predict)
ridge_accuracy=accuracy_score(ridge_predict.round(0),y4_test)
print(RMSE)
print(ridge_r2)
print(ridge_accuracy)


# # We conclude that we have to include case of flush = 0 even though it represents less than 1% of the data, because the score and the accuracy have decreased After testing  , we are gonna choose ridge as it has the best score and accuracy,and it is a parametric model, so now we are going to see the type of errors it does

# In[124]:


# y3_preds is an array, so y3_test must be also an array so that we can compare them
y3_test_array=np.array(y3_test)


# In[125]:


outcome_ridge=[]
for i in y3_test_array:
    if round(y3_preds[i])>=y3_test_array[i]:
        outcome_ridge.append('forecast is higher')
    else:
        outcome_ridge.append('actual is higher')
print(outcome_ridge)


# In[126]:


percentage_ridge=outcome_ridge.count('forecast is higher')/len(outcome_ridge)
print(percentage_ridge)


# # 85% of the predictions are higher than actual, which is good, because we should avoid having a lower forecast value that actual

# # Try to improve model using randomizedsearchcv and gridsearchcv

# In[127]:


model3.get_params()


# In[128]:


group_work_shuffled= group_work.sample(frac=1)
# split into X and y
np.random.seed(42)
X= group_work_shuffled.drop(group_work_shuffled.columns[[0,1,2,3,16,17,18,19]], axis=1)
y= group_work_shuffled['Case of flush']
# split the data into train, validation and test sets
train_split = round(0.7*len(group_work_shuffled)) #70% of data
valid_split = round(train_split +0.15*len(group_work_shuffled )) #15% of data
X_train,  y_train = X[:train_split], y[:train_split]
X_valid, y_valid = X[train_split:valid_split], y[train_split:valid_split]
X_test , y_test = X[valid_split:], y[valid_split:]
len(X_train), len(X_valid), len(X_test)
model4 = Ridge()
model4.fit(X_train, y_train)
# Make baseline predictions
y_preds = model4.predict(X_valid)
# Evaluate the classifier on the validaiton set


# In[129]:


from sklearn.model_selection import RandomizedSearchCV
grid = {'copy_X' :[True,False], #numbers are based on researches
       "max_iter":[None,5,10,20,30,40,50,100,200,300,400],
       'solver':['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
       'normalize': [True,False],
       'alpha':[1,2,4,5],
       'random_state':[None,5,10,15,20,25,30,40,50,70,100],
       'tol':[0.001,0.002,0.003,0.005,0.01,0.015,0.1]}

#split into X and y
np.random.seed(42)
X=group_work_shuffled.drop(group_work_shuffled.columns[[0,1,2,3,16,17,18,19]], axis=1)
y=group_work_shuffled['Case of flush']
# split into train and test set

X_test, X_train , y_test,y_train = train_test_split(X,y, test_size=0.25)
# Initiate RandomForestClassifier
clf =Ridge() #n_jobs choose how much computer cpu dedicate  to ml model
# setup randomized searched cv
rs_clf= RandomizedSearchCV(estimator=clf,
                          param_distributions=grid,
                          n_iter=10, #number of models to try
                          cv=5, # means cross validation
                          verbose=2) 
# fit the randomizedsearchCV version of clf
rs_clf.fit(X_train, y_train);


# In[130]:


rs_clf.best_params_


# In[131]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , mean_squared_error, r2_score
ridge_predict=rs_clf.predict(X_test)

ridge_accuracy=accuracy_score(ridge_predict.round(0),y_test)

ridge_accuracy


# In[132]:


grid_2= {'copy_X' :[False], #numbers are based on researches
       "max_iter":[100,200],
       'solver':['sag'],
       'normalize': [False],
       'alpha':[3,4],
       'random_state':[30,40],
       'tol':[0.001]}


# In[133]:


from sklearn.model_selection import GridSearchCV, train_test_split

#split into X and y
X=group_work_shuffled.drop(group_work_shuffled.columns[[0,1,2,3,16,17,18,19]], axis=1)
y=group_work_shuffled['Case of flush']
np.random.seed(42)
# split into train and test set
X_test, X_train , y_test,y_train = train_test_split(X,y, test_size=0.25)
# Initiate RandomForestClassifier
clf =Ridge() #n_jobs choose how much computer cpu dedicate  to ml model
# setup GridSearchCV
gs_clf= GridSearchCV(estimator=clf,
                          param_grid=grid_2, 
                          cv=5, # means cross validation
                          verbose=2) 
# fit the GridsearchCV version of clf
gs_clf.fit(X_train, y_train); #gs means greadserch
#this is used to try different combination of parameters and figure out the best combination of parameters to maximise accuracy of the Ml MODEL


# In[134]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score , mean_squared_error, r2_score
ridge_predict=gs_clf.predict(X_test)

ridge_accuracy=accuracy_score(ridge_predict.round(0),y_test)

ridge_accuracy


# In[135]:


ridge2= Ridge()
ridge2_params= {'alpha': [0,0.5,1]}
ridge2_grid = GridSearchCV(ridge2, ridge2_params, cv=5, verbose=10 , scoring='neg_mean_absolute_error')
ridge2_grid.fit(X3_train,y3_train)
ridge2_score=ridge2_grid.cv_results_
ridge2_score


# # 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




