#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# In[6]:


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


# # OLS

# In[8]:


X= sm.add_constant(X)
model=sm.OLS(y,X)
result= model.fit()                                                                                                                                                                                                             
print(result.summary())


# ### it reminds us that there exist the problem of multicollinearity

# ## in order to check the multicollinearity between all variables, we can use the variance inflation factor (vif)

# In[11]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

df=flush.drop(['Case of flush','Test'],axis=1)
df=df.dropna(axis=0,how='any')
df['const']=1
x=np.array(df)
vif_list = [variance_inflation_factor(x, i) for i in range(x.shape[1])]
df_vif = pd.DataFrame({'variable': list(df.columns), 'vif': vif_list})
df_vif = df_vif[~(df_vif['variable'] == 'const')]  # remove constance
print(df_vif) #VIF


# ### it proves that there exists high level of multicolllinearity (normally VIF should smaller than 10)

# In[ ]:




