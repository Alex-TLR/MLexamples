''' 
Author: Aleksej Avramovic
Date: 09/08/2023

Cross validation using stratified train/test split.
All classifiers use default parameters.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from classifiers import linearRegression
from classifiers import logisticRegression
from classifiers import decisionTree
from classifiers import supportVectorMachine
from classifiers import knn 
from classifiers import randomForest
from classifiers import gaussianNB
from classifiers import FCN
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn import cross_validation 


# Import data
data = pd.read_csv("creditcard.csv")

# Check the number of valid and fraud transactions
validTransactions = data[data['Class'] == 0]
fraudTransactions = data[data['Class'] == 1]
print(f'Valid cases: {len(validTransactions)}')
print(f'Fraud cases: {len(fraudTransactions)}')
print("\n")

# Useful features are V1, V2 ... V28 and Amount
# The output is Class
X = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Make stratified train/test split
skf = StratifiedKFold(n_splits = 5, shuffle = False)
for train, test in skf.split(X, y):
    print("\n")
    print(train)
    print(train.shape)
    validTransactionsTrain = sum(y[train] == 0)
    fraudTransactionsTrain = sum(y[train] == 1)
    # print(validTransactionsTrain)
    # print(fraudTransactionsTrain)
    print(f'Valid train cases: {validTransactionsTrain}')
    print(f'Fraud train cases: {fraudTransactionsTrain}')
    print(test)
    print(test.shape)
    validTransactionsTest = sum(y[test] == 0)
    fraudTransactionsTest = sum(y[test] == 1)
    print(f'Valid test cases: {validTransactionsTest}')
    print(f'Fraud test cases: {fraudTransactionsTest}')
    print("\n")


scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

LogRegression = linear_model.LogisticRegression()
out1 = cross_validate(LogRegression, X, y, cv = skf, scoring = scores)
print("Logistic regression:")
AP = out1['test_average_precision']
map = np.average(AP)
print(f'mAP is: {map:.3f}\n')

decisionTree = tree.DecisionTreeClassifier()
out2 = cross_validate(decisionTree, X, y, cv = skf, scoring = scores)
print("Decision trees:")
AP = out2['test_average_precision']
map = np.average(AP)
print(f'mAP is: {map:.3f}\n')

kNN = KNeighborsClassifier()
out3 = cross_validate(kNN, X, y, cv = skf, scoring = scores)
print("k Nearest Neighbors:")
AP = out3['test_average_precision']
map = np.average(AP)
print(f'mAP is: {map:.3f}\n')

randomForestClassifier = RandomForestClassifier()
out4 = cross_validate(randomForestClassifier, X, y, cv = skf, scoring = scores)
print("Random forest:")
AP = out4['test_average_precision']
map = np.average(AP)
print(f'mAP is: {map:.3f}\n')