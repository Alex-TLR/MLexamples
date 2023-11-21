''' 
Author: Aleksej Avramovic
Date: 16/10/2023

Wine quality prediction using different classifiers.
Training and testing data from CSV files are concatenated and later splited into train/test ratio
All classifiers use default parameters.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
# from classifiers import linearRegression, logisticRegression, decisionTree, randomForest, knn, supportVectorMachine, gaussianNB
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import tree
from sklearn.model_selection import cross_validate, cross_val_score

# Encoders
ohe = OneHotEncoder()
ord = OrdinalEncoder()

# Import both training and testing data
data = pd.read_csv("winequality-red.csv")

print("Data info")
print(data.info())
print("\n")

# print("First few entries to the dataset:")
# print(data.head())
# print("\n")

# print("Data description:")
# print(data.describe())
# print("\n")

# Uncomment if needed 
# print("Number of null elements")
# print(data.isnull().sum())
# print("\n")

# Plot dependecies if needed
# data.hist(bins = 20, figsize = (10, 10))
# plt.show()

# plt.bar(data['quality'], data['alcohol'])
# plt.xlabel('quality')
# plt.ylabel('alcohol')
# plt.show()

# plt.bar(data['quality'], data['sulphates'])
# plt.xlabel('quality')
# plt.ylabel('sulphates')
# plt.show()

# plt.bar(data['quality'], data['pH'])
# plt.xlabel('quality')
# plt.ylabel('pH')
# plt.show()

# plt.bar(data['quality'], data['density'])
# plt.xlabel('quality')
# plt.ylabel('density')
# plt.show()

# plt.bar(data['quality'], data['total sulfur dioxide'])
# plt.xlabel('quality')
# plt.ylabel('total sulfur dioxide')
# plt.show()

# plt.bar(data['quality'], data['free sulfur dioxide'])
# plt.xlabel('quality')
# plt.ylabel('free sulfur dioxide')
# plt.show()

# plt.bar(data['quality'], data['chlorides'])
# plt.xlabel('quality')
# plt.ylabel('chlorides')
# plt.show()

# plt.bar(data['quality'], data['residual sugar'])
# plt.xlabel('quality')
# plt.ylabel('residual sugar')
# plt.show()

# plt.bar(data['quality'], data['citric acid'])
# plt.xlabel('quality')
# plt.ylabel('citric acid')
# plt.show()

# plt.bar(data['quality'], data['volatile acidity'])
# plt.xlabel('quality')
# plt.ylabel('volatile acidity')
# plt.show()

# plt.bar(data['quality'], data['fixed acidity'])
# plt.xlabel('quality')
# plt.ylabel('fixed acidity')
# plt.show()

print(data.corr())

# plt.figure(figsize=(12, 12))
# sb.heatmap(data.corr() > 0.7, annot=True, cbar=False)
# plt.show()

# Number of unique output values
outputs = data['quality'].unique()
print("outputs\n", outputs)
# There are six different outputs, i.e. quality levels ranging from 3 to 8

print("Total")
unique_values, counts = np.unique(data['quality'], return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")
print("\n")

# The output is Class
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = data['quality'])
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("Train")
unique_values, counts = np.unique(y_train, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")
print("\n")
print("Test")
unique_values, counts = np.unique(y_test, return_counts=True)
for value, count in zip(unique_values, counts):
    print(f"{value} occurs {count} times")


# X = X.reshape(-1, 1)
# print(X.shape)
# X_ord = ord.fit_transform(X).ravel()
# print(X_ord.shape)
# X_ohe = ohe.fit_transform(X).toarray()
# print(X_ohe.shape)
# x_train_ord = ord.fit_transform(x_train).ravel()
# x_test_ord = ord.fit_transform(x_test).ravel()
# x_train_ohe = ohe.fit_transform(x_train).toarray()
# x_test_ohe = ohe.fit_transform(x_test).toarray()

y = y.reshape(-1, 1)
# print(y.shape)
y_ord = ord.fit_transform(y).ravel()
# print(y_ord.shape)
y_ohe = ohe.fit_transform(y).toarray()
# print(y_ohe.shape)
y = y.ravel()
# print(y)
y_train_ord = ord.fit_transform(y_train).ravel()
y_test_ord = ord.fit_transform(y_test).ravel()
y_train_ohe = ohe.fit_transform(y_train).toarray()
y_test_ohe = ohe.fit_transform(y_test).toarray()


# scores = ['average_precision']
# scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']

scores = ['average_precision', 'precision_macro']

# LogRegression = linear_model.LogisticRegression(multi_class = 'ovr', max_iter = 500)
# # scores = cross_validate(LogRegression, X, y_ohe, error_score='raise', scoring = scores, cv = 5)
# scores = cross_validate(LogRegression, X, y_ohe, scoring = scores, cv = 2)

# print(scores)
      
# # Linear regression
# linearRegression.linearRegressionMulticlass(x_train, x_test, y_train_ohe, y_test_ohe)

# # Logistic regression
# logisticRegression.logisticRegressionMulticlass(x_train, x_test, y_train_ord, y_test_ord)
# # ap1 = logisticRegression.logisticRegressionCV(X, y_ord)
# # map1 = np.average(ap1)

# Decision trees
# decisionTree.decisionTreeMulticlass(x_train, x_test, y_train_ord, y_test_ord)
dtree = tree.DecisionTreeClassifier()

blah = cross_validate(dtree, X, y_ohe, scoring = scores, cv = 2)
print(blah)

# # Decision trees
# # ap2 = decisionTree.decisionTreeCV(X, y)
# # map2 = np.average(ap2)
# # print(f'mAP is: {map2:.3f}\n')

# # # Gaussian Navie Bayes
# gaussianNB.GaussianNaiveBayesMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# # KNN
# knn.kNearestNeighborsMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# # Random forest
# randomForest.randomForestMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# # support Vector Machine
# supportVectorMachine.supportVectorMachineLinearMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# supportVectorMachine.supportVectorMachineRbfMulticlass(x_train, x_test, y_train_ord, y_test_ord)

