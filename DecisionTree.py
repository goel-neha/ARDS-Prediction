# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 09:02:28 2017

@author: neha
"""


#Reading files
import os
import random as rn
import pandas as pd
import numpy as np
import sklearn as sk
import pdb
from matplotlib import pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier


# Defines the location of the Files
loc = "/Users/nehansh/Documents/Machine_Learning_Class/ardsfinalproject/Ards/ncui/yes"
loc1 = "/Users/nehansh/Documents/Machine_Learning_Class/ardsfinalproject/Ards/ncui/no"
loc2 = "/Users/nehansh/Documents/Machine_Learning_Class/ardsfinalproject/Ards/tncui/yes"
loc3 = "/Users/nehansh/Documents/Machine_Learning_Class/ardsfinalproject/Ards/tncui/no"


txtfilesTrain = []   #array to contain training files
#Function to add label for specific loctaion of the training text files
def listTrainfiles(a, b):
    count = 0
    for file in os.listdir(a):
        try:
            if file.endswith(".txt"):
                txtfilesTrain.append([a + "/" + str(file), b])
                count = count + 1
            else:
                print ("There is no text file")
        except Exception as e:
            raise e
            print ("No Files found here!")
    print ("Total files found:", count )



txtfilesTest = []
#Function to add label for specific loctaion of the test text files
def listTestfiles(a, b):
    count = 0
    for file in os.listdir(a):
        try:
            if file.endswith(".txt"):
                txtfilesTest.append([a + "/" + str(file), b])
                count = count + 1
            else:
                print ("There is no text file")
        except Exception as e:
            raise e
            print ("No Files found here!")
    print ("Total files found:", count )

# calls the above function according to the location and assigns them labels
listTrainfiles(loc, 1)
listTrainfiles(loc1, 0)
listTestfiles(loc2, 1)
listTestfiles(loc3, 0)

#Shuffle the txtfilesTrain and txtfilesTest array
rn.shuffle(txtfilesTrain)
rn.shuffle(txtfilesTest)

#Extract the data from above arrays through data frame and pandas
df_train = pd.DataFrame(txtfilesTrain, columns = ['filepath', 'label'])
df_test = pd.DataFrame(txtfilesTest, columns = ['filepath', 'label'])

# X & Y sets for each train and test filepaths and labels
df_filepathTrain = df_train['filepath']
df_filepathTest = df_test['filepath']
y_targetTrain = df_train['label']
y_targetTest = df_test['label']

# these documents reads and stores the string code of each files
Train_document = [open(one_document, "r").read() for one_document in df_filepathTrain]

# the X and y for the training data
X = np.asarray(Train_document)
y = np.asarray(y_targetTrain)
# print (X)

#Pipeline for fiting the algorithm & vectorizer on trainining data
pipe_svc = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', DecisionTreeClassifier(random_state=1)),
])

#Grid Search Implementation
# param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__criterion': ['gini', 'entropy'],
              'clf__min_samples_split': [2, 10, 20],
              'clf__max_depth': [None, 2, 5, 10],
              'clf__min_samples_leaf': [1, 5, 10],
              'clf__max_leaf_nodes': [None, 5, 10, 20],
              }]


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=1)

# pdb.set_trace()

# K-fold Implementation
kf = KFold(n_splits=10) # Define the split - into 10 folds
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
# print(kf)
scores = []
for train_index, test_index in kf.split(X):
    # print (k)
    # print('TRAIN:', train_index, 'TEST:', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # print (X_train, y_train)
    pipe_svc.fit(X[train_index], y[train_index])
    score = pipe_svc.score(X[test_index], y[test_index])
    scores.append(score)
    print ('The Scores for each fold : ')
    print(score)

# print (X_train)

#fitting grid search after k-fold split
gs = gs.fit(X_train, y_train)
print ('The Best Score after fitting the Grid Search')
print(gs.best_score_)
print ('The best parameters of the Grid search :')
print(gs.best_params_)

# fitting the best estimator on train set
clf = gs.best_estimator_
clf.fit(X_train, y_train)

#Working on Test Data

#Reading the Test Data Files
Test_document = [open(onedocument, "r").read() for onedocument in df_filepathTest]

#the X and Y for the Test data
X_targetTest = np.asarray(Test_document)
y_targetTest = np.asarray(y_targetTest)

predictedClf = clf.predict(X_targetTest)

# # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# # print(confmat)

print('Accuracy: %.3f' % accuracy_score(y_true=y_targetTest, y_pred=predictedClf))
print(metrics.classification_report(y_targetTest, predictedClf))
print(metrics.confusion_matrix(y_targetTest, predictedClf))



print("ROC Curve")

fpr, tpr, thresholds = roc_curve(y_targetTest, predictedClf, pos_label=1)
print('False Positive Rate:', fpr)
print('True Positive Rate: ', tpr)
#print('thresholds:', thresholds)
AUC = auc(fpr, tpr)
plt.figure(1)
plt.title('Receiver Operating Characteristic For Decision Tree')
plt.plot(fpr, tpr)
plt.plot([0,1], [0, 1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
print(AUC)

# plt.figure(2)
# plt.title('Scatter Plot')
# plt.scatter(y_targetTest, prediction, color = 'blue')
# plt.show
