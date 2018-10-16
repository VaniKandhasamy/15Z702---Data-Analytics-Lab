
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np 
df1 = pd.read_csv("Z:\data analytics lab\IRIS.csv",encoding="latin-1") 
print(df1) 


# In[3]:


df1["Species"] = df1["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)
print(df1)


# In[4]:


x = df1[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df1['Species']
print(x)


# In[5]:


print(y)


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y)


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit only to the training data
scaler.fit(x_train)


# In[8]:


x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# In[9]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))
mlp.fit(x_train,y_train)


# In[10]:


predictions = mlp.predict(x_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))


# In[11]:


print(classification_report(y_test,predictions))

