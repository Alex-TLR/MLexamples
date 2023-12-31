''' 
Author: Aleksej Avramovic
Date: 24/07/2023

Try to optmize Random Forest performance.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from classifiers import randomForest 

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

# Try different number of estimators
nErrors = []
for i in range(10, 200, 10):
    predictions = randomForest.rfNumberOfEstimators(x_train, x_test, y_train, i)
    result = (predictions != y_test).sum()
    print(result)
    nErrors.append(result)    
numOfEstimators = nErrors.index(min(nErrors))*10 + 10
print(f'Best number of neighbors is {numOfEstimators}.')

print("Best performance:")
randomForest.randomForestOptimized(x_train, x_test, y_train, y_test, numOfEstimators, 'binary')