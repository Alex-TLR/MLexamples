''' 
Author: Aleksej Avramovic
Date: 13/08/2023

Disease prediction using different classifiers.
Training and testing data from CSV files are concatenated and later splited into train/test ratio
All classifiers use default parameters.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from classifiers import linearRegression, logisticRegression, decisionTree, randomForest, knn, supportVectorMachine, gaussianNB
from sklearn.model_selection import train_test_split

# Encoders
ohe = OneHotEncoder()
ord = OrdinalEncoder()

# Import both training and testing data
data1 = pd.read_csv("Training.csv").dropna(axis = 1)
data2 = pd.read_csv("Testing.csv").dropna(axis = 1)

data = pd.concat([data1, data2], axis = 0)
print(data1.info())
print(data2.info())
print(data.info())

# The output is Class
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 42) #shuffle = False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
y_train_ord = ord.fit_transform(y_train).ravel()
y_test_ord = ord.fit_transform(y_test).ravel()
y_train_ohe = ohe.fit_transform(y_train).toarray()
y_test_ohe = ohe.fit_transform(y_test).toarray()


# Linear regression
linearRegression.linearRegressionMulticlass(x_train, x_test, y_train_ohe, y_test_ohe)

# Logistic regression
logisticRegression.logisticRegressionMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# Decision trees
decisionTree.decisionTreeMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# Gaussian Navie Bayes
gaussianNB.GaussianNaiveBayesMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# KNN
knn.kNearestNeighborsMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# Random forest
randomForest.randomForestMulticlass(x_train, x_test, y_train_ord, y_test_ord)

# support Vector Machine
supportVectorMachine.supportVectorMachineLinearMulticlass(x_train, x_test, y_train_ord, y_test_ord)
