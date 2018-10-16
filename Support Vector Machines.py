from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

#load data set
irisdataset = datasets.load_iris()

X = irisdataset.data
y = irisdataset.target

# Support vector machines

# Linear svc

# Add noisy features
random_state = np.random.RandomState(0)
n_samples, n_features = X.shape
X = np.c_[X, random_state.randn(n_samples, 100 * n_features)]

#split int training and testing data
X_train, X_test, y_train, y_test = train_test_split(X [y < 2], y [y < 2], test_size = 0.20) 

C=1.0
#linear svc
linearsvc=svm.SVC(kernel='linear')
linearsvc.fit(X_train,y_train)
svmpred=linearsvc.predict(X_test)
#accuracy and confusion matrix
print("Accuracy: "+str(accuracy_score(y_test, svmpred)))  
print("Confusion matrix: \n"+str(confusion_matrix(y_test, svmpred))) 
print("Classification report: \n"+str(metrics.classification_report(y_test,svmpred)))

# precison-recall curve for linearsvc


y_score = linearsvc.decision_function(X_test)
precision, recall , _  = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')


# Polynomial kernel

polysvc=svm.SVC(kernel='poly',random_state=27)
polysvc.fit(X_train,y_train)
svmpred=polysvc.predict(X_test)
#accuracy and confusion matrix
print("Accuracy: "+str(accuracy_score(y_test, svmpred)))  
print("Confusion matrix: \n"+str(confusion_matrix(y_test, svmpred))) 
print("Classification report: \n"+str(metrics.classification_report(y_test,svmpred)))


# Precision - recall curve for polynomial kernel


y_score = polysvc.decision_function(X_test)
precision, recall , _  = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')


# Radial Basis function as kernel


rbfsvc=svm.SVC(kernel='rbf')
rbfsvc.fit(X_train,y_train)
svmpred=rbfsvc.predict(X_test)
#accuracy and confusion matrix
print("Accuracy: "+str(accuracy_score(y_test, svmpred)))  
print("Confusion matrix: \n"+str(confusion_matrix(y_test, svmpred))) 
print("Classification report: \n"+str(metrics.classification_report(y_test,svmpred)))


# Precision - recall curve for radial basis function as kernel

y_score = rbfsvc.decision_function(X_test)
precision, recall , _  = precision_recall_curve(y_test, y_score)
plt.step(recall, precision, color='r')
plt.xlabel('Recall')
plt.ylabel('Precision')

