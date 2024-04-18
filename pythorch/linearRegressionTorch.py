''' 
Author: Aleksej Avramovic
Last update: 18/04/2024

Inspired by: Deep Learning with PyTorch Live Course (freeCodeCamp.org)

Linear regression implementation from the scratch.
Classification of transactions using different classifiers.
In this case, a transaction can be valid or fraud, therefore it is binary classification.
All classifiers use default parameters.
The shuffle for train/test split is disabled. Every new run should give the same results.

'''
import sys
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Utility functions

def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
    percent = f'{100 * (iter / (float(total))):.1f}'
    filled = int(length * iter // total)
    bar = fill * filled + '_' * (length - filled) + ' ' + percent
    sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
    sys.stdout.flush()

def model_batch(x, w, b, bs):
    bb = b * torch.ones(bs, 1)
    return x @ w.t() + bb

def mse(pred, target):
    diff = pred - target
    return torch.sum(diff * diff) / diff.numel()



# Load Fraud detection data
data = pd.read_csv("../001_CreditCardFraudDetection/creditcard.csv")
X = data.iloc[:, 1:30].values
y = data.iloc[:, -1].values

# Standardize data
print("Scale the data.")
scaler = StandardScaler()
X = scaler.fit_transform(X)

X = np.float32(X)
y = np.float32(y)

X = torch.from_numpy(X)
y = torch.from_numpy(y)

# Batch size
batchSize = 64

# Learning rate
lr = 1e-5

# Number of epochs
numberOfEpochs = 5

# Split to train/test data and basic analytics
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = False)
print("Number of test cases:", len(y_test))
numberOfTestCases = len(y_test)
validTransactionsTest = y_test[y_test == 0]
fraudTransactionsTest = y_test[y_test == 1]
print(f'Valid test cases: {len(validTransactionsTest)}')
print(f'Fraud test cases: {len(fraudTransactionsTest)}')
print("\n")

# Implementation from the scratch.
print("From the scratch!")

w = torch.randn(1, X.shape[1], requires_grad = True)
b = torch.randn(1, requires_grad = True)

print(w)
print(b)
print(f'Mean of weights is {torch.mean(w).item():.4f}, and the standard deviation is {torch.std(w).item():.4f}')

# Train one epoch
numberOfBatches = int(y.shape[0]/batchSize)

for n in range(numberOfEpochs):
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

    progressBar(n + 1, numberOfEpochs, prefix = 'Progress: ', suffix = 'Loss: ' + f'{loss.item():.2f}', length = 60, fill = '#')

print("\n")

# Validate on the test set
y_test = y_test.unsqueeze(1)
y_val = x_test @ w.t() + b * torch.ones(numberOfTestCases, 1)
# Compare with y_test
test_diff = y_test - y_val
MSE = torch.sum(test_diff * test_diff) / test_diff.numel()
print(f'Mean sqaure error is {MSE:.5f}.')

print(f'Thresholding (in order to achive 0 or 1 output).')
y_pred = torch.threshold(y_val, 0.5, 0)
test_diff = y_test - y_pred
MSE = torch.sum(test_diff * test_diff) / test_diff.numel()
print(f'Mean sqaure error is {MSE:.5f}.')

# Compare 
comparison1 = torch.not_equal(y_test, y_pred)
numberOfMisMatches = torch.sum(comparison1).item()
comparison2 = torch.eq(y_test, y_pred)
numberOfMatches = torch.sum(comparison2).item()

# t1 = y_pred.detach().numpy()
# t2 = y.detach().numpy()
# file_path_pred = "predictions.txt"
# np.savetxt(file_path_pred, t1, fmt = '[%d]', delimiter = '\n')
# file_path_ground = "ground.txt"
# np.savetxt(file_path_ground, t2, fmt = '[%d]', delimiter = '\n')

print(f'Number of matches is {numberOfMatches}')
print(f'Number of mis-matches is {numberOfMisMatches}')
print("\n")


# Implementation using PyTorch built-ins
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as f

print("Using built-ins.")

# Define dataset
data = TensorDataset(x_train, y_train)

train_data = DataLoader(data, batchSize, shuffle=False)

# Define the model
model = nn.Linear(29, 1)
print(model.weight)
print(model.bias)
print(f'Mean of weights is {torch.mean(model.weight).item():.4f}, and the standard deviation is {torch.std(model.weight).item():.4f}')

# Define loss
lossFunction = f.mse_loss

# Define optimization
opt = torch.optim.SGD(model.parameters(), lr = lr)

def fit(epochs, model, lossFunction, opt, train_data):
    '''
    This is the fit function that requiers the follwoing:
    epochs:       number of epoch for training
    model:        the model that is trained
    lossFunction: the function to calculate the loss in each iteration
    opt:          an optimization function 
    train_data:   training data loader object
    '''

    for i in range(epochs):
        for x_train, y_train in train_data:
            # Generate prediction
            pred = model(x_train)
            # Calculate loss
            y_train = y_train.unsqueeze(1)
            loss = lossFunction(pred, y_train)
            # Compute gradients
            loss.backward()
            # Update parameters
            opt.step()
            # Reset gradients
            opt.zero_grad()

        progressBar(i + 1, epochs, prefix = 'Progress: ', suffix = 'Loss: ' + f'{loss.item():.2f}', length = 60, fill = '#')
    
fit(numberOfEpochs, model, lossFunction, opt, train_data)
print("\n")

# Validate on the test set
# y_test = y_test.unsqueeze(1)
y_val = model(x_test)
# Compare with y_test
test_diff = y_test - y_val
MSE = torch.sum(test_diff * test_diff) / test_diff.numel()
print(f'Mean sqaure error is {MSE:.5f}.')

y_pred = torch.threshold(y_val, 0.5, 0)
test_diff = y_test - y_pred
MSE = torch.sum(test_diff * test_diff) / test_diff.numel()
print(f'Mean sqaure error is {MSE:.5f}.')

# Compare 
comparison1 = torch.not_equal(y_test, y_pred)
numberOfMisMatches = torch.sum(comparison1).item()
comparison2 = torch.eq(y_test, y_pred)
numberOfMatches = torch.sum(comparison2).item()

print(f'Number of matches is {numberOfMatches}')
print(f'Number of mis-matches is {numberOfMisMatches}')