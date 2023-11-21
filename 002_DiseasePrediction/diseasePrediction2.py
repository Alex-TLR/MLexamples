''' 
Author: Aleksej Avramovic
Date: 13/08/2023

The second experiment on disease prediction data:
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
print("Print data info:")
print(data1.info())
print(data2.info())
print(data.info())

# The output is Class
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# print(X.shape)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0) #shuffle = False)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

print("Print shape of the data:")
print("x_train shape: ", x_train.shape)
print("x_test shape: ", x_test.shape)
print("y_train shape: ", y_train.shape)
print("y_test shape: ", y_test.shape)
print("\n")

y_train_ord = ord.fit_transform(y_train).ravel()
y_test_ord = ord.fit_transform(y_test).ravel()
y_test_ord = y_test_ord.ravel()
y_train_ohe = ohe.fit_transform(y_train).toarray()
y_test_ohe = ohe.fit_transform(y_test).toarray()

# Linear regression
linearRegression.linearRegressionClassification(x_train, x_test, y_train_ohe, y_test_ohe, 'multi')

# Logistic regression
logisticRegression.logisticRegressionClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')

# Decision trees
decisionTree.decisionTreeClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')

# # Gaussian Navie Bayes
# gaussianNB.GaussianNaiveBayesClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')

# # KNN
# knn.kNearestNeighborsClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')

# # Random forest
# randomForest.randomForestClassification(x_train, x_test, y_train_ord, y_test_ord, 'multi')

# # support Vector Machine
# supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train_ord, y_test_ord, 'linear', 'multi')
