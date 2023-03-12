#!/usr/bin/env python
# coding: utf-8

# In[121]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.linear_model import LogisticRegression


# In[122]:


flush= pd.read_csv("flush.csv", header = 0, sep = ",")
X = flush.iloc[:, 4:16].values
y = flush.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)


# In[123]:


pca=PCA(n_components=2)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_ratio_)


# we can see that keeping 2 components have and explained ratio of around 90%.

# In[124]:


#Create feature vector with 2 components measure above
X_pca = pca.fit_transform(X)
X_pca = pd.DataFrame(data = X_pca , columns = ['Feature vector 1', 'Feature vector 2'])
finalX = pd.concat([X_pca, flush[["Case of flush"]]], axis = 1)


# In[125]:


print(X_pca)


# Choosing 2 as components number is already not bad, but we can still dig deeper to choose the best components number through visualization

# In[126]:


pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


# Accroding to the line graph we can choose 4 as components number

# In[127]:


#pca fit with 4 components
pca=PCA(n_components=4)
pca.fit(X)
X_pca = pca.transform(X)
X_pca = pd.DataFrame(data = X_pca , columns = ['Feature vector 1', 'Feature vector 2', 'Feature vector 3', 'Feature vector 4'])


# We can verify if there still exists comultilinearity with Variance Inflation Factor (VIF)

# In[128]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
#constance
X_pca['constance']=1
a=np.array(X_pca)
vif_list = [variance_inflation_factor(a,i) for i in range (a.shape[1])]
#define a dataframe to show vif result
vif_df = pd.DataFrame({'variable components': list(X_pca.columns),'VIF':vif_list})
vif_df = vif_df[~(vif_df['variable components']=='constance')]

print(vif_df)


# It shows that there's no comultilinearity among these variables

# ## linear regression with pca

# In[129]:


#use these variable to fit linear regression model
#drop constance
X_pca= X_pca.drop(labels='constance',axis=1)
X1_train, X1_test, y1_train, y1_test = train_test_split(X_pca,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)


# In[130]:


model1=LinearRegression()
model1.fit(X1_train,y1_train)
print(model1.intercept_)
print(model1.coef_)

linear_predict = model1.predict(X1_test)
print(mean_squared_error(y1_test,linear_predict))
print(r2_score(y1_test,linear_predict))
print(accuracy_score(y1_test,linear_predict.round(0)))


# ## confusion matrix

# In[116]:


cm=confusion_matrix(y1_test, linear_predict.round(0))

sns.heatmap(cm, fmt = '.2e', cmap = 'GnBu')
plot.show()


# ## ridge regression with pca

# In[117]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X_pca,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)


# In[118]:


alphas = 10**np.linspace(-3,3,100)
ridge_cofficients = []

for alpha in alphas:
    ridge = Ridge(alpha = alpha, normalize=True)
    ridge.fit(X2_train, y2_train)
    ridge_cofficients.append(ridge.coef_)
    
# CV of ridge regression
ridge_cv = RidgeCV(alphas = alphas, normalize=True, cv = 10)
ridge_cv.fit(X2_train, y2_train)
# retrieve bet lambda value
ridge_best_alpha = ridge_cv.alpha_
print(ridge_best_alpha)


# In[137]:


# model fit base on lambda=ridge_best_alpha
ridge = Ridge(alpha = ridge_best_alpha, normalize=True)
ridge.fit(X2_train, y2_train)
# coef of ridge regression
ridge.coef_

# Prediction
ridge_predict = ridge.predict(X2_test)
# Evaluation
RMSE = np.sqrt(mean_squared_error(y2_test,ridge_predict))
ridge_r2 = r2_score(y2_test,ridge_predict)
ridge_accuracy=accuracy_score(y2_test,ridge_predict.round(0))
ridge_accuracy1=accuracy_score(y2_test,ridge_prediction.round(0))
print(RMSE)
print(ridge_r2)
print(ridge_accuracy)


# accuracy ridge without PCA = 0.4193

# ## Lasso regression with PCA

# In[120]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X_pca,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)


# In[50]:


alphas = 10**np.linspace(-4,4,100)
lasso_cofficients = []

for alpha in alphas:
    lasso = Lasso(alpha = alpha, normalize=True, max_iter=10000)
    lasso.fit(X3_train, y3_train)
    lasso_cofficients.append(lasso.coef_)
    
#cv
# LASSO regression cross validation
lasso_cv = LassoCV(alphas = alphas, normalize=True, cv = 10, max_iter=10000)
lasso_cv.fit(X3_train, y3_train)
# best alpha
lasso_best_alpha = lasso_cv.alpha_
lasso_best_alpha


# In[138]:


# lasso regression using best alpha value
lasso = Lasso(alpha = lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X3_train, y3_train)

# predict
lasso_predict = lasso.predict(X3_test)
# score
RMSE = np.sqrt(mean_squared_error(y3_test,lasso_predict))
lasso_r2 = r2_score(y3_test,lasso_predict)
lasso_accuracy=accuracy_score( y3_test,lasso_predict.round(0))
print(RMSE)
print(lasso_r2)
print(lasso_accuracy)


# accuracy lasso without pca = 0.4838709677419355

# ## Logistic regression with pca

# In[107]:


X4_train, X4_test, y4_train, y4_test = train_test_split(X_pca,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)


# In[108]:


import warnings
warnings='ignore'

logistic=LogisticRegression(C=100, class_weight='balanced', max_iter=50,
                   multi_class='multinomial', solver='newton-cg')
logistic.fit(X4_train, y4_train)


# In[109]:


# prediction
y4_pred = logistic.predict(X4_test)
acu = accuracy_score(y4_test, y4_pred)  # accuracy
r2 = r2_score(y4_test,y4_pred)
print('accuracy= ',acu)
print('r2 score= ',r2)


# In[110]:


y_pred_array=np.array(y4_pred)
y_test_array=np.array(y4_test)
R=np.array([y_pred_array, y_test_array])

outcome=[]
for val in np.nditer(R):
    if y_pred_array[val] >= y_test_array[val] :
        outcome.append('forecast higher')
    else:
        outcome.append('actual higher')
 

print(outcome)


# In[111]:


percentage = outcome.count('forecast higher')/len(outcome)
print(percentage)


# In[ ]:




