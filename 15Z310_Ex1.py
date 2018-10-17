
# coding: utf-8

# In[1]:


import pandas as pd
df = pd.read_csv('EmployeeAttrition.csv')
print(df.shape)
df.head(5)


# In[2]:


df.describe()


# In[3]:


print(len(df))
df.isnull().sum()


# In[7]:


import matplotlib.pyplot as plt
df.boxplot('WorkLifeBalance')


# ###### A boxplot between 'WorkLifeBalance' and 'JobLevel' is drawn.There is a positive correlation between the two attributes. If an employee has a good work life balance then they are higher up in the job level.

# In[12]:


plt.scatter(df.TotalWorkingYears,df.JobLevel,s=df.Age)

