import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# import linearRegression
# import logisticRegression
from classifiers import decisionTree
# import supportVectorMachine
# import knn 
from classifiers import randomForest

ohe = OneHotEncoder()

# Import training data
data = pd.read_csv("Training.csv")
print(data.info())

# Import test data
dataTest = pd.read_csv("Testing.csv")

# Print some data description
print(data.shape)
print(data.describe())

# We need to check how many different prognisis we have.
numOfPrognosis = data['prognosis'].nunique()   # pd.unique(data['prognosis'])
print(f'Unique values for prognosis is {numOfPrognosis}.')


# Useful features are itching, skin_rash, nodal_skin_eruptions, continuous_sneezing... etc
# The output is prognosis (converted with One Hot Encoding)

# Make input and putput data
x_train = data.iloc[:, :132].values
y_train = data['prognosis'].to_numpy()
x_test = dataTest.iloc[:, :132].values
y_test = dataTest['prognosis'].to_numpy()
# print(x.shape)
# print(y.shape)
# print(y)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(y.shape)
y_train = ohe.fit_transform(y_train).toarray()
y_test = ohe.fit_transform(y_test).toarray()
# print(y.shape)


# Normalize data
# I am not sure that we need to normalize

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

# # Linear regression
# linearRegression.linearRegression(x_train, x_test, y_train, y_test)

# # Logistic regression
# logisticRegression.logisticRegression(x_train, x_test, y_train, y_test)

# # Decision trees
decisionTree.decisionTree(x_train, x_test, y_train, y_test)

# # KNN
# knn.kNearestNeighbors(x_train, x_test, y_train, y_test)

# # Random forest
randomForest.randomForest(x_train, x_test, y_train, y_test)

# # support Vector Machine
# supportVectorMachine.supportVectorMachine(x_train, x_test, y_train, y_test)
