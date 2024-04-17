''' 
Author: Aleksej Avramovic
Date: 21/11/2023

Wine quality prediction using different classifiers.
Training and testing data from CSV files are concatenated and later splited into train/test ratio
All classifiers use default parameters.
Remove most correlated features
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from classifiers import linearRegression, logisticRegression, decisionTree, randomForest, knn, supportVectorMachine, gaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sb; sb.set(font_scale = 1.2)

# Different quality distribution
def mapQuality(x):
    if x < 5:
        return 0
    if x == 5:
        return 1
    if x == 6:
        return 2
    if x > 6:
        return 3
    

# Encoders
ohe = OneHotEncoder()
ord = OrdinalEncoder()

# Import both training and testing data
data = pd.read_csv("winequality-red.csv")

print("Data info")
print(data.head())
print("\n")


# plt.figure(figsize=(12, 12))
# sb.heatmap(data.corr() > 0.3, annot=True, cbar=False)
# plt.show()

# data = data.drop('fixed acidity', axis=1)
# data = data.drop('free sulfur dioxide', axis=1)
# data = data.drop('density', axis=1)
# data = data.drop('sulphates', axis=1)
# print(data.corr())

# plt.figure(figsize=(12, 12))
# sb.heatmap(data.corr() > 0.3, annot=True, cbar=False)
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


# Make different quality distribution
# quality 3 and 4 are LOW
# quality 5 is LOW MEDIUM
# quality 6 is HIGH MEDIUM
# quality 7 and 8 are HIGH
data['new quality'] = [mapQuality(i) for i in data.quality]

print(data.head(20))

data.drop('quality', inplace = True, axis = 1)

print(data.head(20))

# The output is Class
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = y.reshape(-1, 1)

# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = data['quality'])
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle=False)

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
linearRegression.linearRegressionClassification(x_train, x_test, y_train_ohe, y_test_ohe, 'multi')

# Logistic regression
logisticRegression.logisticRegressionClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
# p1 = logisticRegression.logisticRegressionCV(X, y_ord, 'multi')
# ap1 = np.average(p1)
# print(f'Macro average precision is: {ap1:.3f}\n')

# Decision trees
decisionTree.decisionTreeClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
# p2 = decisionTree.decisionTreeCV(X, y_ord, 'multi')
# ap2 = np.average(p2)
# print(f'Macro average precision is: {ap2:.3f}\n')

# Gaussian Navie Bayes
gaussianNB.GaussianNaiveBayesClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
# p3 = gaussianNB.GaussianNaiveBayesCV(X, y_ord, 'multi')
# ap3 = np.average(p3)
# print(f'Macro average precision is: {ap3:.3f}\n')

# KNN
knn.kNearestNeighborsClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
# p4 = knn.kNearestNeighborsCV(X, y_ord, 'multi')
# ap4 = np.average(p4)
# print(f'Macro average precision is: {ap4:.3f}\n')

# Random forest
randomForest.randomForestClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')
# p5 = randomForest.randomForestCV(X, y_ord, 'multi')
# ap5 = np.average(p5)
# print(f'Macro average precision is: {ap5:.3f}\n')

# support Vector Machine
supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train_ord, y_test_ord, 'linear', 'multi')
supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train_ord, y_test_ord, 'rbf', 'multi')
# p6 = supportVectorMachine.supportVectorMachineCV(X, y_ord, 'rbf', 'multi')
# ap6 = np.average(p6)
# print(f'Macro average precision is: {ap6:.3f}\n')