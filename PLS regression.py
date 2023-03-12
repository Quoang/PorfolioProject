#!/usr/bin/env python
# coding: utf-8

# In[85]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plot
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[86]:


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


# ## Partial least squares regression can be used when there is colinearity among variables

# In[87]:


from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV


# In[88]:


#set up model and parameter
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 4)}

#GridSearchCV to adjust parameter
gsearch = GridSearchCV(pls_model_setup, param_grid)

#train fit
pls_model = gsearch.fit(X_train, y_train)


# ## model evaluation

# In[89]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[92]:


#print coef
print('Partial Least Squares Regression coefficients:',pls_model.best_estimator_.coef_)

#prediction
pls_prediction = pls_model.predict(X_test)
pls_pred = pls_prediction+0.5

#evaluation
pls_acu= accuracy_score(y_test,pls_prediction.round(0))
pls_acu1= accuracy_score(y_test,pls_pred.round(0))

print('accuracy=',pls_acu)
print(pls_acu1)


# Accuracy is rather low

# In[91]:


cm=confusion_matrix(y_test, y_pred)

sns.heatmap(cm, fmt = '.2e', cmap = 'GnBu')
plot.show()


# In[ ]:





# In[ ]:





# In[ ]:




