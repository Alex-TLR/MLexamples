''' 
Author: Aleksej Avramovic
Date: 10/07/2023

Try to optmize Decision Tree performance.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classifiers import decisionTree

methodList = ["gini", "entropy", "log_loss"]

# Import data
data = pd.read_csv("creditcard.csv")

# Check the number of valid and fraud transactions
validTransactions = data[data['Class'] == 0]
fraudTransactions = data[data['Class'] == 1]
print(f'Valid cases: {len(validTransactions)}')
print(f'Fraud cases: {len(fraudTransactions)}')

# Useful features are V1, V2 ... V28 and Amount
# The output is Class
x = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values
print(x.shape)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)
print("Number of test samples", len(y_test))
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Decision trees with different criterions
nErrors = []
for i in range(len(methodList)):
    predictions = decisionTree.dtCriterion(x_train, x_test, y_train, methodList[i])
    result = (predictions != y_test).sum()
    # print(result)
    nErrors.append(result)

ind = nErrors.index(min(nErrors))
c = methodList[ind]
print(f'Best criterion is {c}.')

nErrors = []
for i in range(1, 10):
    predictions = decisionTree.dtMaxDepth(x_train, x_test, y_train, c, i)
    result = (predictions != y_test).sum()
    # print(result)
    nErrors.append(result)    

maxDepth = nErrors.index(min(nErrors)) + 1
print(f'Best max_depth is {maxDepth}.')

nErrors = []
for i in range(5, 100, 5):
    predictions = decisionTree.dtMaxLeafNodes(x_train, x_test, y_train, c, maxDepth, i)
    result = (predictions != y_test).sum()
    # print(result)
    nErrors.append(result)    

maxLeafNodes = (nErrors.index(min(nErrors)) + 1)*5
print(f'Best max_leaf_nodes is {maxLeafNodes}.')

print("Best performance:")
decisionTree.dtOptimized(x_train, x_test, y_train, y_test, c, maxDepth, maxLeafNodes)