
# coding: utf-8

# In[9]:


import pandas as pd
df=pd.read_csv('Z:\EmployeeAttrition.csv')
df


# In[10]:


df.describe


# In[11]:


df.sum()


# In[12]:


df.mean()


# In[13]:


df.median() 


# In[14]:


df.count() 


# In[15]:


df.min()


# In[16]:


df.std()


# In[17]:


df.isnull().any()


# In[19]:


df.boxplot(figsize='10,10')


# In[20]:


df.boxplot(figsize='20,20') 


# In[22]:


import matplotlib.pyplot as p
p.boxplot(df['DistanceFromHome'])
p.show() 


# In[23]:


import matplotlib.pyplot as p
p.boxplot(df['TotalWorkingYears']) 
p.show() 


# In[24]:


import matplotlib.pyplot as p
p.boxplot(df['MonthlyIncome']) 
p.show() 


# In[25]:


df.boxplot(figsize='6,5') 

