
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score


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
df_train = pd.DataFrame(txtfilesTrain, columns = ['filepath', 'label']).astype(str)
df_test = pd.DataFrame(txtfilesTest, columns = ['filepath', 'label']).astype(str)

# X & Y sets for each train and test filepaths and labels
df_filepathTrain = df_train['filepath'].astype(str)
df_filepathTest = df_test['filepath'].astype(str)
y_targetTrain = df_train['label']
y_targetTest = df_test['label']

# these documents reads and stores the string code of each files
Train_document = [open(one_document, "r").read() for one_document in df_filepathTrain]
Test_document = [open(onedocument, "r").read() for onedocument in df_filepathTest]


# the X and y for the training data
X = np.asarray(Train_document)
y = np.asarray(y_targetTrain)
# print (X)

# for test data
X_Test = np.asarray(Test_document)
y_Test = np.asarray(y_targetTest)

#Pipeline for fiting the algorithm & vectorizer on trainining data
pipe_svc = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier()),
])

#Grid Search Implementation
param_range = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [{'clf__alpha': param_range, 'clf__loss': ['squared_hinge'],'clf__penalty': ['l2'], 'clf__random_state':[1]},
               {'clf__alpha': param_range, 'clf__loss': ['hinge'],'clf__penalty': ['l2'], 'clf__random_state':[1]}]


gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=10,
                  n_jobs=-1)

gs = gs.fit(X, y)
print("Grid Search Best Score:")
print(gs.best_score_)
print()
print("Grid Search Best Params:")
print(gs.best_params_)

clf = gs.best_estimator_

clf.fit(X, y)

print()
print('Test accuracy: %.3f' % clf.score(X_Test, y_Test))

prediction = clf.predict(X_Test)
#print(prediction)
accuracy = np.mean(prediction == y_Test)
print(accuracy)
print()
print("Classification Report:")
print(metrics.classification_report(y_Test, prediction))
print()
print("Confusion Matrix:")
print(metrics.confusion_matrix(y_Test, prediction))

"""

scores = cross_val_score(gs, X, y, scoring='accuracy')
print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
"""
# pdb.set_trace()
"""
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
    print(score)

# print (X_train)
"""
#fitting grid search after k-fold split
"""
print(gs.best_score_)
print(gs.best_params_)

# fitting the best estimator on train set
clf = gs.best_estimator_
clf.fit(X, y)

#Calculating accuracy on Validation Data Set
print('Test accuracy: %.3f' % clf.score(X_test, y_test))
# print('\nCV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))
"""
"""
#Working on Test Data

#Reading the Test Data Files
Test_document = [open(onedocument, "r").read() for onedocument in df_filepathTest]

#the X and Y for the Test data
X_targetTest = np.asarray(Test_document)
y_targetTest = np.asarray(y_targetTest)

vect = CountVectorizer()
X_test_counts = vect.transform(Test_document)
tf_test_transform = TfidfTransformer(use_idf = False).fit(X_test_counts)
X_test_tf = tf_test_transform.transform(X_test_counts)
predictedClf = clf.predict(X_test_tf)

# # confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
# # print(confmat)

print('Accuracy: %.3f' % accuracy_score(y_true=y_targetTest, y_pred=predictedClf))
print('Precision: %.3f' % precision_score(y_true=y_targetTest, y_pred=predictedClf))
print('Recall: %.3f' % recall_score(y_true=y_targetTest, y_pred=predictedClf))
print('F1: %.3f' % f1_score(y_true=y_targetTest, y_pred=predictedClf))
"""
