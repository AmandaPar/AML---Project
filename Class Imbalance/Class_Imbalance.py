# -*- coding: utf-8 -*-
"""
Created on Sun May 09 10:58:34 2021

@author: Amanda Parpori
"""

# ---------------------------------- Class Imbalance ------------------------------
#                                    Prepare Dataset
# ---------------------------------------------------------------------------------

from sklearn import ensemble
import numpy as np
import pandas as pd

cardio = pd.read_excel('cardio.xlsx', engine = 'openpyxl')

# check for null values per column
print('\nCheck for null values:\n')
print(cardio.isnull().sum())

# fill NA values
cardio.fillna( method ='ffill', inplace = True) 

print('\nCheck frequency distribution of target value:')
print(cardio['cardio'].value_counts())

# make the dataset imbalanced with ratio 1:10

cardio.sort_values('cardio', inplace=True)

range_to_del = list(range(35021,66500))

cardio.drop(cardio.index[range_to_del], inplace=True)

print('\nFrequency distribution after modification:')
print(cardio['cardio'].value_counts())

print("\n")
print(cardio.isnull().sum())

# make the dataset imbalanced with ratio 1:100

range_to_del = list(range(35021,38171))

cardio2 = cardio.drop(cardio.index[range_to_del])

print('\nFrequency distribution after modification 2:')
print(cardio2['cardio'].value_counts())


# ---------------------------------- Class Imbalance ------------------------------
# ---------------------------------------------------------------------------------

import warnings
warnings.filterwarnings('ignore')
from numpy import mean
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from imblearn.ensemble import EasyEnsembleClassifier

from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression

X = cardio.drop(columns=['cardio', 'id'])
y = cardio['cardio']


from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter
from sklearn.datasets import  fetch_openml
from sklearn.compose import make_column_transformer, make_column_selector


#-----------------------------------------------------------------------------------
#                                  Unaware LR vs aware LR
#-----------------------------------------------------------------------------------
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve

clf1 = LogisticRegression(solver="lbfgs")
clf2 = LogisticRegression(solver="lbfgs", class_weight="balanced")

scoring = ['accuracy', 'balanced_accuracy']

print("\nLR lbfgs:\n")
scores = cross_validate(clf1, X, y, scoring=scoring, cv=10, return_train_score=False)
for s in scoring:
    print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

print("\nLR lbfgs with balanced weights:\n")
scores = cross_validate(clf2, X, y, scoring=scoring, cv=10, return_train_score=False)
for s in scoring:
    print("%s: %.2f (+/- %.2f)" % (s, scores["test_" + s].mean(), scores["test_" + s].std()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
clf1.fit(X_train, y_train)
plot_roc_curve(clf1, X_test, y_test)
plot_precision_recall_curve(clf1, X_test, y_test)

clf2.fit(X_train, y_train)
plot_roc_curve(clf2, X_test, y_test)
plot_precision_recall_curve(clf2, X_test, y_test)


#-------------------------------------------------------------------------------------
# near miss
#-------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("\n\n-------------------------Near Miss--------------------------\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Create a pipeline
pipeline = make_pipeline(LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("logistic regression without near miss")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR without near miss")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot() 
print("\n\n")



# Create a pipeline
pipeline = make_pipeline(LogisticRegression(solver="lbfgs", class_weight='balanced'))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("logistic regression with balanced weights")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR without near miss with balanced weights\n")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print("\n\n")



# Create a pipeline
pipeline = make_pipeline(NearMiss(version=1),
                         LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("logistic regression with near miss 1")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR with near miss 1\n")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print("\n\n")



# Create a pipeline
pipeline = make_pipeline(NearMiss(version=2),
                         LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("logistic regression with near miss 2")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR with near miss 2")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print("\n\n")



# Create a pipeline
print("logistic regression with near miss 3")
pipeline = make_pipeline(NearMiss(version=3, n_neighbors_ver3=3),
                         LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR with near miss 3")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print("\n\n")




#--------------------------------------------------------------------------------
# synthetic oversampling
#--------------------------------------------------------------------------------

print("\n\n------------------------------Synthetic Oversampling--------------------------\n")

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced
from imblearn.ensemble import EasyEnsembleClassifier
import matplotlib.pyplot as plt


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# Create a pipeline
pipeline = make_pipeline(LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("logistic regression")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

# Create a pipeline
pipeline = make_pipeline(LogisticRegression(solver="lbfgs", class_weight='balanced'))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("\n\nlogistic regression with balanced weights")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))

# Create a pipeline
pipeline = make_pipeline(SMOTE(random_state=3, k_neighbors=5),
                         LogisticRegression(solver="lbfgs"))
pipeline.fit(X_train, y_train)

# Classify and report the results
print("\n\nlogistic regression with SMOTE")
print(classification_report_imbalanced(y_test, pipeline.predict(X_test)))
print("Conf. Matrix of LR with SMOTE")
cm = confusion_matrix(y_test, pipeline.predict(X_test))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()












