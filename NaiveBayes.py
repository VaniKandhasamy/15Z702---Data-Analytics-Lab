
# coding: utf-8

# In[1]:


from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
from sklearn.metrics import classification_report,confusion_matrix 
from sklearn.model_selection import train_test_split 


# In[2]:


iris = datasets.load_iris() 
X = iris.data
y = iris.target 


# In[3]:


# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3 , random_state=0) 
gnb = GaussianNB() 


# In[4]:


y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test) 
cnf_matrix_gnb = confusion_matrix(y_test, y_pred_gnb) 


# In[5]:


print(cnf_matrix_gnb) 


# In[6]:


from sklearn.metrics import accuracy_score 

print(accuracy_score(y_test, y_pred_gnb)) 


# In[8]:


from sklearn.metrics import precision_score
print(precision_score(y_test, y_pred_gnb,average='weighted')) 


# In[9]:


print(classification_report(y_test,y_pred_gnb)) 

