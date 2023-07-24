''' 
Author: Aleksej Avramovic
Date: 24/07/2023

Disease prediction using different classifiers.
All classifiers use default parameters.
'''

import sys
sys.path.append('../')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from classifiers import linearRegression, logisticRegression, decisionTree, randomForest, knn, supportVectorMachine

# Encoders
ohe = OneHotEncoder()
ord = OrdinalEncoder()

# Import training data
data = pd.read_csv("Training.csv")
# print(data.info())

# Import test data
dataTest = pd.read_csv("Testing.csv")

# Print some data description
# print(data.shape)
# print(data.describe())

# We need to check how many different prognisis we have.
numOfPrognosis = data['prognosis'].nunique()   
print(f'Unique values for prognosis is {numOfPrognosis}.')

prognosis = dataTest['prognosis']
print(prognosis)

# Useful features are itching, skin_rash, nodal_skin_eruptions, continuous_sneezing... etc
# The output is prognosis (converted with One Hot Encoding for Linear regression, or Ordinar Encoder for others)

# Make input and putput data
x_train = data.iloc[:, :132].values
y_train = data['prognosis'].to_numpy()
x_test = dataTest.iloc[:, :132].values
y_test = dataTest['prognosis'].to_numpy()
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_train_ord = ord.fit_transform(y_train).ravel()
y_test_ord = ord.fit_transform(y_test).ravel()
y_train_ohe = ohe.fit_transform(y_train).toarray()
y_test_ohe = ohe.fit_transform(y_test).toarray()

# Linear regression
linearRegression.linearRegression(x_train, x_test, y_train_ohe, y_test_ohe, 1)

# Logistic regression
logisticRegression.logisticRegression(x_train, x_test, y_train_ord, y_test_ord)

# Decision trees
decisionTree.decisionTree(x_train, x_test, y_train_ord, y_test_ord)

# KNN
knn.kNearestNeighbors(x_train, x_test, y_train_ord, y_test_ord)

# Random forest
randomForest.randomForest(x_train, x_test, y_train_ord, y_test_ord)

# support Vector Machine
supportVectorMachine.supportVectorMachineLinear(x_train, x_test, y_train_ord, y_test_ord)
