#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score,confusion_matrix, accuracy_score, recall_score,classification_report
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score as AUC


# In[65]:


flush= pd.read_csv("flush.csv", header = 0, sep = ",")
X = flush.iloc[:, 4:16].values
y = flush.iloc[:, 18].values
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=2)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# We choose multinomial as multi-class parameter to get a better result. We will test all the 3 possible solvers the find the best one. (liblinear is not possible with multinomial class)

# In[66]:


log_model =LogisticRegression(multi_class='multinomial',penalty='l2', solver='sag',max_iter=10000)
log_model.fit(X_train,y_train)


# In[67]:


#model evaluation with 'newton-cg'
y_pred = log_model.predict(X_test)
acu = accuracy_score(y_test, y_pred)  # accuracy
recall = recall_score(y_test, y_pred, average="macro")  # recall
r2 = r2_score(y_test,y_pred)
print(y_pred)
print('accuracy= ',acu)
print('recall= ',recall)
print('r2 score= ',r2)


# ### use gridsearch CV to improve accuracy

# In[76]:


from sklearn.model_selection import GridSearchCV
import warnings


# Setting parameters
params = {'multi_class':['multinomial','ovr'],
          'C':[0.0001, 1, 100, 1000],
          'max_iter':[1, 10, 50, 100],
          'class_weight':['balanced', None],
          'solver':['liblinear','sag','lbfgs','newton-cg']
         }
lr = LogisticRegression()
clf = GridSearchCV(lr, param_grid=params, cv=10)
clf.fit(X_train,y_train)


# In[77]:


classifier = LogisticRegression(**clf.best_params_)
# fit
classifier.fit(X_train, y_train)


# In[78]:


# prediction
log2_pred = classifier.predict(X_test)
acu = accuracy_score(y_test, log2_pred)  # accuracy
recall = recall_score(y_test, log2_pred, average="macro")  # recall
r2 = r2_score(y_test,log2_pred)
print('accuracy= ',acu)
print('recall= ',recall)
print('r2 score= ',r2)


# ## confusion matrix

# In[81]:


cm=confusion_matrix(y_test, log2_pred)

sns.heatmap(cm, fmt = '.2e', cmap = 'GnBu')
plt.show()


# In[84]:


print(log2_pred,y_test)


# In[105]:


y_pred_array=np.array(log2_pred)
y_test_array=np.array(y_test)
R=np.array([y_pred_array, y_test_array])


# In[106]:


outcome=[]
for val in np.nditer(R):
    if y_pred_array[val] >= y_test_array[val] :
        outcome.append('forecast higher')
    else:
        outcome.append('actual higher')
 

print(outcome)


# In[107]:


percentage = outcome.count('forecast higher')/len(outcome)
print(percentage)


# In[108]:


for val in np.nditer(R):
    print(y_pred_array[val]-y_test_array[val])


# In[ ]:




