#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import pandas and scikit libraries
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import tree
#For Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
import numpy as np

import matplotlib.pyplot as plt

import pydot
import pydotplus
import collections
#For Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB


# In[2]:


#read the dataset and convert to dataframe
heart_data = pd.read_csv('cardio_train.csv', sep = ';')

#print the top 5 rows
heart_data.head()


# In[3]:


#print the bottom 5 rows
heart_data.tail()


# In[4]:


heart_data.info()
#print the class labels
heart_data['cardio'].tolist()


# In[5]:


#Split your dataset 70% for training, and 30% for testing the classifier
X_train, X_test, y_train, y_test = train_test_split(heart_data.iloc[:, :12], heart_data.iloc[:, 12], test_size=0.30, random_state=42)


# In[6]:


# Decision tree classification using Gini
D_tree = DecisionTreeClassifier(criterion = 'gini', max_depth=3, random_state=67)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
param_grid = {'max_depth': [2, 4, 6, 8, 10]}
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
grid = GridSearchCV(D_tree, param_grid, cv=cv, return_train_score = True, scoring='accuracy')
grid.fit(X_train, y_train)


# In[43]:


#Printing the accuracy for decision tree classification using Gini
from sklearn.metrics import accuracy_score
y_train_hat  = grid.predict(X_train)
y_test_hat  = grid.predict(X_test)
in_sample_acc = accuracy_score(y_train,y_train_hat) * 100
out_of_sample_acc = accuracy_score(y_test,y_test_hat) * 100
#print("In-sample Accuracy: ", in_sample_acc)
print("Out-of-sample Accuracy: ", out_of_sample_acc)


# In[44]:


#Printing the confusion matrix and classification report for decision tree classification using Gini
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix: \n",confusion_matrix(y_test,y_test_hat))
print("Classification Report: \n",classification_report(y_test,y_test_hat))


# In[45]:


#Printing the decision tree with Gini values
D_tree.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(22, 8))
tree.plot_tree(D_tree, fontsize=13, feature_names = heart_data.columns[:-1], filled = True)
plt.show()


# In[46]:


#Decision tree classification using Entropy
D_tree = DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=67)
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
param_grid = {'max_depth': [2, 4, 6, 8, 10]}
cv = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
grid = GridSearchCV(D_tree, param_grid, cv=cv, return_train_score = True, scoring='accuracy')
grid.fit(X_train, y_train)


# In[47]:


#Printing the accuracy for decision tree classification using Entropy
from sklearn.metrics import accuracy_score
y_train_hat  = grid.predict(X_train)
y_test_hat  = grid.predict(X_test)
in_sample_acc = accuracy_score(y_train,y_train_hat, normalize = True) * 100
out_of_sample_acc = accuracy_score(y_test,y_test_hat, normalize = True) * 100
#print("In-sample Accuracy: ", in_sample_acc)
print("Out-of-sample Accuracy: ", out_of_sample_acc)


# In[48]:


#Printing the confusion matrix and classification report for decision tree classification using Entropy
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix: \n",confusion_matrix(y_test,y_test_hat))
print("Classification Report: \n",classification_report(y_test,y_test_hat))


# In[49]:


#Printing the decision tree with Entropy values
D_tree.fit(X_train, y_train)
fig, ax = plt.subplots(figsize=(22, 8))
tree.plot_tree(D_tree, fontsize=13, feature_names = heart_data.columns[:-1], filled = True)
plt.show()


# In[50]:


#Comparing Gini and Entropy
import matplotlib.pyplot as plt
plt.bar(['Gini','Entropy'], [73.29, 73.30], align='center', alpha=0.6, width=0.5, color = 'red')
plt.xticks(['Gini','Entropy'],fontsize=15)
plt.ylabel('Accuracies')
plt.title('Decision Tree')
plt.show()


# In[51]:


#Naive Bayes Classification of data
naive=GaussianNB()
naive = naive.fit(X_train,y_train)

#Predict the response for test dataset
y_hat = naive.predict(X_test)

#Printing Accuracy
out_of_sample_acc = accuracy_score(y_test,y_hat, normalize = True) * 100
print("Out-of-sample Accuracy: ", out_of_sample_acc)

#Printing confusion matrix and classification report
print("Confusion Matrix: \n",confusion_matrix(y_test,y_hat))
print("Classification Report: \n",classification_report(y_test,y_hat))

