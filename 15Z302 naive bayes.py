
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#load the csv file
df=pd.read_csv("Z:/data analytics lab/IRIS.csv",encoding="latin-1") 

#Replace Species type with numbers
df.Species.replace(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'], [1, 2, 3], inplace=True)

clf = GaussianNB()

# Split-out validation dataset
array = df.values
X = array[:,1:5]
Y = array[:,5]

# One-third of data as a part of test set
validation_size = 0.33

seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

scoring = 'accuracy'

#Fitting the training set
clf.fit(X_train, Y_train) 
#Predicting for the Test Set
pred_clf = clf.predict(X_validation)
prob_pos_clf = clf.predict_proba(X_validation)[:, 1]
pred_clf_df = pd.DataFrame(pred_clf.reshape(50,1))
pred_clf_df.rename(columns={0:'Prediction'}, inplace=True)
X_validation_df = pd.DataFrame(X_validation.reshape(50,4))

#concatenating the two pandas dataframes over the columns to create a prediction dataset
pred_outcome = pd.concat([X_validation_df, pred_clf_df], axis=1, join_axes=[X_validation_df.index])

pred_outcome.rename(columns = {0:'SepalLengthCm', 1:'SepalWidthCm', 2:'PetalLengthCm', 3:'PetalWidthCm'}, inplace=True)

del df['Id']

#merging the prediction with original dataset
pred_comp = pd.merge(df,pred_outcome, on=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

#print top 10 lines of the final predictions
print((pred_comp).head(10))
print ("\n")

#Save the file to csv
pred_comp.to_csv('Predictions.csv', sep=',')

#Model Performance
#setting performance parameters
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kfold, scoring=scoring)

#displaying the mean and standard deviation of the prediction
msg = "%s: %f (%f)" % ('NB accuracy', cv_results.mean(), cv_results.std())
print(msg)

