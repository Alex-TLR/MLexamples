import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import linearRegression
import logisticRegression
import decisionTree
import supportVectorMachine
import knn 
import randomForest

# Import data
data = pd.read_csv("creditcard.csv")
# data.head() #check this out

# Print some data description
# print(data.shape)
# print(data.describe())

# Check the number of valid and fraud transactions
validTransactions = data[data['Class'] == 0]
fraudTransactions = data[data['Class'] == 1]
print(f'Valid cases: {len(validTransactions)}')
print(f'Fraud cases: {len(fraudTransactions)}')

# Useful features are V1, V2 ... V28 and Amount
# The output is Class

x = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Normalize data
# x = (x - np.mean(x))/np.std(x)

# Standardize data
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
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
supportVectorMachine.supportVectorMachine(x_train, x_test, y_train, y_test)
