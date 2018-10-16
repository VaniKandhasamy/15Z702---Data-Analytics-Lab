
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris
i= load_iris() 
i['data'].shape 


# In[2]:


X = i['data'] 

y = i['target'] 


# In[3]:


from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, y) 

from sklearn.preprocessing import StandardScaler 

scaler = StandardScaler() 

# Fit only to the training data 

scaler.fit(X_train) 


# In[4]:


X_train = scaler.transform(X_train) 

X_test = scaler.transform(X_test) 

from sklearn.neural_network import MLPClassifier 

mlp = MLPClassifier(hidden_layer_sizes=(4,4,4),max_iter=500) 

mlp = MLPClassifier(hidden_layer_sizes=(4,4,4,4),max_iter=500) 


# In[7]:


predictions = mlp.predict(X_test) 
from sklearn.metrics import classification_report,confusion_matrix 
 
print(confusion_matrix(y_test,predictions)) 


# In[8]:


print(classification_report(y_test,predictions)) 

