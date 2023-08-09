''' 
Author: Aleksej Avramovic
Date: 26/07/2023

Try to optmize linear SVM performance.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from classifiers import supportVectorMachine 

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

# Try different values of C
nErrors = []
listC = [0.001, 0.01, 0.1, 1, 10, 100]
kernel = 'rbf'
for i in listC:
    predictions = supportVectorMachine.svmC(x_train, x_test, y_train, kernel, i)
    result = (predictions != y_test).sum()
    print(result)
    nErrors.append(result)    
C = listC[nErrors.index(min(nErrors))]
print(f'Best value of C is {C}.')

print("Best performance:")
supportVectorMachine.svmOptimized(x_train, x_test, y_train, y_test, kernel, C, 'binary')