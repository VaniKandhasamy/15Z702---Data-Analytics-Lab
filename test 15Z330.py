
# coding: utf-8

# In[1]:


from scipy.io import arff
import pandas as pd


# In[9]:


data = arff.loadarff(r"C:\Users\cse30\Desktop\Sapfile1.arff")
df = pd.DataFrame(data[0])
df.head()


# # basic description on data

# In[11]:


print(df.index)


# In[12]:


print(df.columns)


# In[13]:


df.describe()


# In[16]:


print(df)


# In[15]:


df.sample()


# # visualizing data

# In[38]:


import matplotlib.pyplot as plt
import numpy as np
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2000', periods=1000))
ts = ts.cumsum()
ts.plot()




# # identify patterns

# In[45]:


raw_data = {'fo': ['Farmer'], 
        'mq': ['112']}
df1 = pd.DataFrame(raw_data, columns = ['fo', 'mq'])
df1


# # Descriptive statistics

# In[24]:


df.sum()


# In[26]:


df.mean()


# In[27]:


df.count()


# In[28]:


df.min()


# In[35]:


df.isnull().any()


# In[33]:


df.boxplot(figsize='1,5') 


# # identifying challenges

# In[36]:


df.isnull().any()


# In[46]:


df.plot(x='atd', y='tt', style='o')

