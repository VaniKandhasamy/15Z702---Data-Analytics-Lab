
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import svm
x= pd.read_csv("Z:/data analytics lab/IRIS.csv",encoding="latin-1") 
print(x)


# In[2]:


x["Species"] = x["Species"].map({
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2
}).astype(int)
print(x)


# In[3]:


a = np.array(x) 
y  = a[:,5] # classes having 0 and 1 
  
# extracting two features 
x = np.column_stack((x.SepalLengthCm,x.SepalWidthCm)) 
x.shape
  
print (x),(y) 


# In[7]:


from sklearn.svm import SVC # "Support Vector Classifier" 
clf = SVC(kernel='linear') 
  
# fitting x samples and y classes 
clf.fit(x, y)


# In[5]:


print(clf.predict([[5.1,3.5]]))


# In[6]:


svc=clf.fit(x, y)
import matplotlib.pyplot as plt
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 np.arange(y_min, y_max, h))
plt.subplot(1, 1, 1)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
plt.show()

