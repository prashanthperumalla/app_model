#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib


# In[ ]:


def predict(data):
    clf=joblib.load("reg.sav")
    return clf.predict(data)

