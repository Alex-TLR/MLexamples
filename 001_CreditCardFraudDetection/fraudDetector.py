''' 
Author: Aleksej Avramovic
Date: 10/07/2023

Classification of transactions using different classifiers.
All classifiers use default parameters.
The shuffle for train/test split is disabled. Every new run should give the same results.
'''

import sys
import os
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from classifiers import linearRegression
from classifiers import logisticRegression
from classifiers import decisionTree
from classifiers import supportVectorMachine
from classifiers import knn 
from classifiers import randomForest

# Import data
data = pd.read_csv("creditcard.csv")

# Print some data description
# print(data.shape)

# Check the number of valid and fraud transactions
validTransactions = data[data['Class'] == 0]
fraudTransactions = data[data['Class'] == 1]
print(f'Valid cases: {len(validTransactions)}')
print(f'Fraud cases: {len(fraudTransactions)}')

# Useful features are V1, V2 ... V28 and Amount
# The output is Class
x = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)
print("Number of test samples", len(y_test))
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Linear regression
linearRegression.linearRegression(x_train, x_test, y_train, y_test)

# Logistic regression
logisticRegression.logisticRegression(x_train, x_test, y_train, y_test)

# Decision trees
decisionTree.decisionTree(x_train, x_test, y_train, y_test)

# KNN
knn.kNearestNeighbors(x_train, x_test, y_train, y_test)

# Random forest
randomForest.randomForest(x_train, x_test, y_train, y_test)

# support Vector Machine
supportVectorMachine.supportVectorMachineLinear(x_train, x_test, y_train, y_test)

supportVectorMachine.supportVectorMachineRbf(x_train, x_test, y_train, y_test)
