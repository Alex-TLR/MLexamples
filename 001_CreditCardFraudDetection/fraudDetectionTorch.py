''' 
Author: Aleksej Avramovic
Last update: 17/04/2024

Linear regression implementation from the scratch.
Classification of transactions using different classifiers.
In this case, a transaction can be valid or fraud, therefore it is binary classification.
All classifiers use default parameters.
The shuffle for train/test split is disabled. Every new run should give the same results.
Tree-based methods can have different outcome from different runs.
'''

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def model_batch(x, w, b, bs):
    bb = b * torch.ones(bs, 1)
    return x @ w.t() + bb

def mse(pred, target):
    diff = pred - target
    return torch.sum(diff * diff) / diff.numel()

# Fraud detection
data = pd.read_csv("C:\MachineLearning\MLexamples\\001_CreditCardFraudDetection\creditcard.csv")
X = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

X = np.float32(X)
y = np.int32(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
print("Number of test cases:", len(y_test))
numberOfTestCases = len(y_test)
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Batch size
batchSize = 64

# Learning rate
lr = 1e-5

X = torch.from_numpy(x_train)
y = torch.from_numpy(y_train)
w = torch.randn(1, X.shape[1], requires_grad = True)
b = torch.randn(1, requires_grad = True)

# Train one epoch
numberOfBatches = int(y.shape[0]/batchSize)
for i in range(numberOfBatches):
    Xb = X[i*batchSize : (i+1)*batchSize, :]
    yb = y[i*batchSize : (i+1)*batchSize]
    yb = yb.unsqueeze(1)
    yPred = model_batch(Xb, w, b, batchSize)
    loss = mse(yPred, yb)
    loss.backward()
    with torch.no_grad():
        w -= w.grad*lr 
        b -= b.grad*lr 
        w.grad.zero_()
        b.grad.zero_()

# Finish the rest of the inputs
if (X.shape[0] > numberOfBatches*batchSize):
    numberOfSamples = X.shape[0] - numberOfBatches*batchSize
    Xb = X[numberOfBatches*batchSize : , :]
    yb = y[numberOfBatches*batchSize :]
    yb = yb.unsqueeze(1)
    yPred = model_batch(Xb, w, b, numberOfSamples)
    loss = mse(yPred, yb)
    with torch.no_grad():
        w -= w.grad*lr 
        b -= b.grad*lr 
        w.grad.zero_()
        b.grad.zero_()

print("Done.")
print(w)
print(b)

# Validate on the test set
X = torch.from_numpy(x_test)
y = torch.from_numpy(y_test)
y = y.unsqueeze(1)
Y_val = X @ w.t() + b * torch.ones(numberOfTestCases, 1)
# Compare with y_test
test_diff = y - Y_val
MSE = torch.sum(test_diff * test_diff) / test_diff.numel()
print(f'Mean sqaure error is {MSE}.')

