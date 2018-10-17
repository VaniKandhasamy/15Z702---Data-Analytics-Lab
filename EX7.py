
# coding: utf-8

# In[21]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[22]:


iris = pd.read_csv("Iris.csv") 


# In[23]:


iris.head(2)


# In[24]:


iris.info()


# In[40]:



from sklearn.cross_validation import train_test_split 

from sklearn import svm 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB  
from sklearn.preprocessing import LabelEncoder


# In[41]:


train, test = train_test_split(iris, test_size = 0.3)


# In[42]:


train_X = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
train_y=train.Species
test_X= test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']] 
test_y =test.Species   


# In[45]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

clf = SVC(kernel = 'linear').fit(train_X,train_y)
clf.predict(train_X)
y_pred = clf.predict(test_X)


cm = confusion_matrix(test_y, y_pred) 

cm_df = pd.DataFrame(cm,
                     index = ['setosa','versicolor','virginica'], 
                     columns = ['setosa','versicolor','virginica'])

plt.figure(figsize=(5.5,4))
sns.heatmap(cm_df, annot=True)
plt.title('SVM Linear Kernel \nAccuracy:{0:.3f}'.format(accuracy_score(test_y, y_pred)))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


# In[28]:


model = svm.SVC() 
model.fit(train_X,train_y) 
prediction=model.predict(test_X) 
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,test_y))


# # Gaussian naive Bayes 

# In[35]:


le = LabelEncoder()
le.fit(iris['Species'])
iris['Species'] = le.transform(iris['Species'])


trainSet, testSet = train_test_split(iris, test_size = 0.33)
print(trainSet.shape)
print(testSet.shape)
print(trainSet.head(3))


# In[36]:



trainData = pd.DataFrame.as_matrix(trainSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
trainTarget = pd.DataFrame.as_matrix(trainSet[['Species']]).ravel()
testData = pd.DataFrame.as_matrix(testSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
testTarget = pd.DataFrame.as_matrix(testSet[['Species']]).ravel()

classifier = GaussianNB()
classifier.fit(trainData, trainTarget)


# In[37]:


predictedValues = classifier.predict(testData)

nErrors = (testTarget != predictedValues).sum()
accuracy = 1.0 - nErrors / testTarget.shape[0]
print("Accuracy: ", accuracy)

