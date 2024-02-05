#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df= pd.read_csv(r"F:\Imarticus Class\Python\Datasets\Auto MPG Reg\Auto MPG Reg.csv")


# In[4]:


df


# In[11]:


df.info()


# In[7]:


#convert Horse Power to Numeric

df.horsepower= pd.to_numeric(df.horsepower,errors="coerce")


# In[10]:


df.horsepower=df.horsepower.fillna(df.horsepower.median())


# In[13]:


y=df.mpg
X=df.drop(['carname','mpg'],axis=1)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


regmodel= LinearRegression().fit(X,y)


# In[16]:


regmodel.score(X,y)


# In[17]:


regpredit= regmodel.predict(X)


# In[18]:


from sklearn.metrics import mean_squared_error


# In[19]:


np.sqrt(mean_squared_error(y,regpredit))


# In[20]:


# for Deployment model need to saved as .pk(Pickle) or .sav(joblib) library


# In[21]:


import joblib


# In[22]:


joblib.dump(regmodel,"reg.sav")


# In[23]:


import io


# In[24]:


get_ipython().run_line_magic('cd', '')


# In[ ]:




