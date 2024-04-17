''' 
Author: Aleksej Avramovic
Date: 24/11/2023

Wine quality prediction using different classifiers.
All classifiers use default parameters.

Categories (wine qualities) with the small value of samples are discarded in this example.
The classification is done for quility values of 5, 6 and 7.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from classifiers import linearRegression, logisticRegression, decisionTree, randomForest, knn, supportVectorMachine, gaussianNB, xgb
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Encoder
ord = OrdinalEncoder()
ohe = OneHotEncoder()

# PCA
pca = PCA(n_components = 7)

# Import both training and testing data
data = pd.read_csv("winequality-red.csv")

print("Data info")
print(data.head())
print("\n")

print("First few entries to the dataset:")
print(data.head())
print("\n")

print("Data description:")
print(data.describe())
print("\n")

# Uncomment if needed 
# print("Number of null elements")
# print(data.isnull().sum())
# print("\n")

# Number of unique output values
outputs = data['quality'].unique()
print("outputs\n", outputs)
# There are six different outputs, i.e. quality levels ranging from 3 to 8

print("Total")
unique_values, counts = np.unique(data['quality'], return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")
print("\n")

data.drop(data[data['quality'] < 5].index, inplace = True)
data.drop(data[data['quality'] > 7].index, inplace = True)
data.reset_index()

# The output is Class
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# pca.fit(X)
# X = pca.transform(X)

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = data['quality'])

print("Train")
unique_values, counts = np.unique(y_train, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")

print("Test")
unique_values, counts = np.unique(y_test, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")
print("\n")

y_ord = ord.fit_transform(y).ravel()
y_train_ord = ord.fit_transform(y_train).ravel()
y_test_ord = ord.fit_transform(y_test).ravel()
y_ohe = ohe.fit_transform(y).toarray()
y_train_ohe = ohe.fit_transform(y_train).toarray()
y_test_ohe = ohe.fit_transform(y_test).toarray()

# Linear regression
print("Linear regression:")
a0 = linearRegression.linearRegressionClassification(x_train, x_test, y_train_ohe, y_test_ohe, 'multi')
print(f'Accuracy is: {a0:.3f}\n')

# Logistic regression
print("Logistic regression:")
a1 = logisticRegression.logisticRegressionClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
print(f'Accuracy is: {a1:.3f}')
p1 = logisticRegression.logisticRegressionCV(X, y_ord, 'multi')
ap1 = np.average(p1)
print(f'Macro average precision is: {ap1:.3f}\n')

# Decision trees
print("Decision trees:")
a2 = decisionTree.decisionTreeClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
print(f'Accuracy is: {a2:.3f}')
p2 = decisionTree.decisionTreeCV(X, y_ord, 'multi')
ap2 = np.average(p2)
print(f'Macro average precision is: {ap2:.3f}\n')

# Gaussian Navie Bayes
print("Gaussian Navie Bayes:")
a3 = gaussianNB.GaussianNaiveBayesClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
print(f'Accuracy is: {a3:.3f}')
p3 = gaussianNB.GaussianNaiveBayesCV(X, y_ord, 'multi')
ap3 = np.average(p3)
print(f'Macro average precision is: {ap3:.3f}\n')

# KNN
print("k Nearest Neighbors:")
a4 = knn.kNearestNeighborsClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
print(f'Accuracy is: {a4:.3f}')
p4 = knn.kNearestNeighborsCV(X, y_ord, 'multi')
ap4 = np.average(p4)
print(f'Macro average precision is: {ap4:.3f}\n')

# Random forest
print("Random forest:")
a5 = randomForest.randomForestClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
print(f'Accuracy is: {a5:.3f}')
p5 = randomForest.randomForestCV(X, y_ord, 'multi')
ap5 = np.average(p5)
print(f'Macro average precision is: {ap5:.3f}\n')

# Support Vector Machine
print("Support Vector Machine:")
# supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train_ord, y_test_ord, 'linear', 'multi')
a6 = supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train_ord, y_test_ord, 'rbf', 'multi')
print(f'Accuracy is: {a6:.3f}')
p6 = supportVectorMachine.supportVectorMachineCV(X, y_ord, 'rbf', 'multi')
ap6 = np.average(p6)
print(f'Macro average precision is: {ap6:.3f}\n')

# # XGBoost
print("XGBoost:")
a7 = xgb.xgboostClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
a7 = print(f'Accuracy is: {a7:.3f}')
p7 = supportVectorMachine.supportVectorMachineCV(X, y_ord, 'rbf', 'multi')
ap7 = np.average(p7)
print(f'Macro average precision is: {ap7:.3f}\n')