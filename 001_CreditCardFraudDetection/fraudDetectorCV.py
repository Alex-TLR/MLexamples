''' 
Author: Aleksej Avramovic
Date: 20/11/2023

Classification of transactions using different classifiers.
Cross-validation analysis.
'''

#TODO Must be updated, averaged precision is possible only for binary classification

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from classifiers import linearRegression
from classifiers import logisticRegression
from classifiers import decisionTree
from classifiers import supportVectorMachine
from classifiers import knn 
from classifiers import randomForest
from classifiers import gaussianNB
from classifiers import FCN

# Import data
data = pd.read_csv("creditcard.csv")

# Print some data description
# print(data.shape)

# Check the number of valid and fraud transactions
# validTransactions = data[data['Class'] == 0]
# fraudTransactions = data[data['Class'] == 1]
# print(f'Valid cases: {len(validTransactions)}')
# print(f'Fraud cases: {len(fraudTransactions)}')
# print("\n")

# Useful features are V1, V2 ... V28 and Amount
# The output is Class
X = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Logistic regression
ap1 = logisticRegression.logisticRegressionCV(X, y, 'binary')
map1 = np.average(ap1)
print(f'mAP is: {map1:.3f}\n')

# Decision trees
ap2 = decisionTree.decisionTreeCV(X, y, 'binary')
map2 = np.average(ap2)
print(f'mAP is: {map2:.3f}\n')

# Decision trees
ap2 = decisionTree.decisionTreeCVOptimal1(X, y)
map2 = np.average(ap2)
print(f'mAP is: {map2:.3f}\n')

# kNN 
ap3 = knn.kNearestNeighborsCV(X, y, 'binary')
map3 = np.average(ap3)
print(f'mAP is: {map3:.3f}\n')

# kNN 
ap3 = knn.kNearestNeighborsCVOptimal1(X, y)
map3 = np.average(ap3)
print(f'mAP is: {map3:.3f}\n')

# Random forest
ap4 = randomForest.randomForestCV(X, y, 'binary')
map4 = np.average(ap4)
print(f'mAP is: {map4:.3}\n')

# linear SVM
ap5 = supportVectorMachine.supportVectorMachineCV(X, y, 'linear', 'binary')
map5 = np.average(ap5)
print(f'mAP is: {map5:.3}\n')

# linear SVM
ap5 = supportVectorMachine.supportVectorMachineCVOptimal1(X, y, 'linear', 0.001)
map5 = np.average(ap5)
print(f'mAP is: {map5:.3}\n')

# rbf SVM
ap6 = supportVectorMachine.supportVectorMachineCV(X, y, 'rbf', 'binary')
map6 = np.average(ap6)
print(f'mAP is: {map6:.3}\n')

# rbf SVM

# FCN model1