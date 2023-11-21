''' 
Author: Aleksej Avramovic
Last update: 16/11/2023

Classification of transactions using different classifiers.
In this case, a transaction can be valid or fraud, therefore it is binary classification.
All classifiers use default parameters.
The shuffle for train/test split is disabled. Every new run should give the same results.
Tree-based methods can have different outcome from different runs.
'''

import sys
sys.path.append('../')

import pandas as pd
from sklearn.model_selection import train_test_split
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
# data is filled from the CSV file
data = pd.read_csv("creditcard.csv")

# Print some data description (uncomment if necessary)
# print(data.shape)

# Check the number of valid and fraud transactions
print('Number of cases:')
validTransactions = data[data['Class'] == 0]
fraudTransactions = data[data['Class'] == 1]
print(f'Valid cases: {len(validTransactions)}')
print(f'Fraud cases: {len(fraudTransactions)}')

# Observing the data, we can notice that the features are 
# given by V1, V2 ... V28 and Amount
# The output is Class
x = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Standardize data
scaler = StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = False)
print("Number of test cases:", len(y_test))
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Linear regression
linearRegression.linearRegressionClassification(x_train, x_test, y_train, y_test, 'binary')

# Logistic regression
logisticRegression.logisticRegressionClassification(x_train, x_test, y_train, y_test, 'binary')

# Decision trees
decisionTree.decisionTreeClassification(x_train, x_test, y_train, y_test, 'binary')

# KNN
knn.kNearestNeighborsClassification(x_train, x_test, y_train, y_test, 'binary')

# Random forest
randomForest.randomForestClassification(x_train, x_test, y_train, y_test, 'binary')

# support Vector Machine
supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train, y_test, 'linear', 'binary')
supportVectorMachine.supportVectorMachineClassification(x_train, x_test, y_train, y_test, 'rbf', 'binary')

# FCN
# FCN.FullConnectedNetwork1 requires the number of input nodes
#    It creates two different hidden layers with 4 fully connected nodes and Relu activation
#    The last layer is 1 node sigmoid 
#    Optimizer is set to Adam
#    Loss is binary_crossentropy
#    Metrics is accuracy
# In this case there are 29 different input features
model = FCN.FullConnectedNetwork1(29)
# Batch size is equal to 128
# Epochs is 10
# MakeFigures is set to 0 (no figures are made)
model.load_weights('../networks/model1.h5')
# model = FCN.TrainFCN(x_train, y_train, model, 0)
FCN.PredictFCN(x_test, y_test, model)