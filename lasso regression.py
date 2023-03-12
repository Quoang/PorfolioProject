#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plot
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix, accuracy_score


# In[19]:


flush= pd.read_csv("flush.csv", header = 0, sep = ",")
X = flush.iloc[:, 4:16].values
y = flush.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# # lasso regression

# ## lambda value visualizatioin

# In[37]:


alphas = 10**np.linspace(-4,4,100)
lasso_cofficients = []

for alpha in alphas:
    lasso = Lasso(alpha = alpha, normalize=True, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_cofficients.append(lasso.coef_)

plot.style.use('ggplot')
plot.plot(alphas, lasso_cofficients)
plot.xscale('log')
plot.axis('tight')
plot.title('relation between alpha and coefficient')
plot.xlabel('Log Alpha')
plot.ylabel('Coefficients')
plot.show()


# ## cross validation

# In[38]:


#cv
# LASSO regression cross validation
lasso_cv = LassoCV(alphas = alphas, normalize=True, cv = 10, max_iter=10000)
lasso_cv.fit(X_train, y_train)
# best alpha
lasso_best_alpha = lasso_cv.alpha_
lasso_best_alpha


# In[39]:


# lasso regression using best alpha value
lasso = Lasso(alpha = lasso_best_alpha, normalize=True, max_iter=10000)
lasso.fit(X_train, y_train)

# predict
lasso_predict = lasso.predict(X_test)
# score
RMSE = np.sqrt(mean_squared_error(y_test,lasso_predict))
lasso_r2 = r2_score(y_test,lasso_predict)
lasso_accuracy=accuracy_score(y_test,lasso_predict.round(0))

print(RMSE)
print(lasso_r2)
print(lasso_accuracy)


# ## confusion matrix

# In[32]:


cm=confusion_matrix(y_test, lasso_predict.round(0))

sns.heatmap(cm, fmt = '.2e', cmap = 'GnBu')
plot.show()


# In[35]:


print(y_test,lasso_predict.round(0))


# In[ ]:




