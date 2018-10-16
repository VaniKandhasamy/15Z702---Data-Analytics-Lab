
# coding: utf-8

# In[3]:


import pandas as pd 
import numpy as np 
df1 = pd.read_csv("Y:/DA LAB/EmployeeAttrition.csv")
print(df1) 


# In[4]:


df1.describe() 


# In[7]:


import matplotlib.pyplot as plt 
df1[['Department','DailyRate']].boxplot(vert=False,by='Department')
plt.subplots_adjust(left=0.25) 
plt.show() 


# In[9]:


df1.boxplot(vert=False,by='Department') 
plt.subplots_adjust(left=0.25) 
plt.show() 


# In[10]:


df2=df1["JobSatisfaction"] 


# In[11]:


print(df2) 


# In[12]:


df1.boxplot() 


# In[13]:


df1=df1.replace(0,np.NaN) 
print(df1.isnull().sum()) 


# In[14]:


print(df1.head(20)) 

