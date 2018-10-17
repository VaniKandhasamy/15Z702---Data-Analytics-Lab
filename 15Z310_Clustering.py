
# coding: utf-8

# In[1]:


#KMeans
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.labels_


# In[2]:


#Agglometric Clustering
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(linkage="ward",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_


# In[3]:


model = AgglomerativeClustering(linkage="complete",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_


# In[4]:


model = AgglomerativeClustering(linkage="average",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_

