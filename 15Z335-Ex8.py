from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
iris = load_iris()
X = iris.data
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.labels_
from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(linkage="ward",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_
model = AgglomerativeClustering(linkage="complete",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_
model = AgglomerativeClustering(linkage="average",n_clusters=3)
singleLinkage = model.fit(X)
model.labels_

