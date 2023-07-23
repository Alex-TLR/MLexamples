''' 
Author: Aleksej Avramovic
Date: 22/07/2023

Try to optmize k Nearest neighbors performance.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classifiers import knn

algoList = ["ball_tree", "kd_tree", "brute"]

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)
print("Number of test samples", len(y_test))
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Decision trees with different criterions
# Try different algorithms
# nErrors = []
# for i in range(len(algoList)):
#     predictions = knn.knnAlgorithm(x_train, x_test, y_train, algoList[i])
#     result = (predictions != y_test).sum()
#     print(result)
#     nErrors.append(result)
# ind = nErrors.index(min(nErrors))
# a = algoList[ind]
# print(f'Best criterion is {a}.')

# Try different distance (Minkowski with different parameter p)
# nErrors = []
# for i in range(1, 5):
#     predictions = knn.knnDistance(x_train, x_test, y_train, a, i)
#     result = (predictions != y_test).sum()
#     print(result)
#     nErrors.append(result)    
# p = nErrors.index(min(nErrors)) + 1
# print(f'Best distance metric is {p}-Minkowski.')

# Try different number of neighbors
nErrors = []
for i in range(1, 25, 2):
    predictions = knn.knnNumberOfNeighbors(x_train, x_test, y_train, i)
    result = (predictions != y_test).sum()
    # print(result)
    nErrors.append(result)    
numOfNeighbors = nErrors.index(min(nErrors))*2 + 1
print(f'Best number of neighbors is {numOfNeighbors}.')

print("Best performance:")
knn.knnOptimized(x_train, x_test, y_train, y_test, numOfNeighbors)