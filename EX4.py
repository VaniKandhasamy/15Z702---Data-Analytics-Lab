
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[34]:


data = pd.read_csv("spam.csv",encoding='latin-1')


# In[35]:


data.head()


# In[36]:


data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
data = data.rename(columns={"v1":"label", "v2":"text"})


# In[37]:


data.tail()


# In[38]:


data.label.value_counts()


# In[39]:


data['label_num'] = data.label.map({'ham':0, 'spam':1})


# In[40]:


data.head()


# # DATASET SPLIT
# 

# In[41]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data["text"],data["label"], test_size = 0.2, random_state = 10)


# In[42]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[43]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_df = vect.transform(X_train)
X_test_df = vect.transform(X_test)


# # Multinomial Naive Bayes

# In[44]:


prediction = dict()
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_df,y_train)


# In[45]:


prediction["Multinomial"] = model.predict(X_test_df)


# In[46]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[47]:


accuracy_score(y_test,prediction["Multinomial"])


# # KNN

# In[48]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_df,y_train)


# In[49]:


prediction["knn"] = model.predict(X_test_df)


# In[50]:


accuracy_score(y_test,prediction["knn"])

