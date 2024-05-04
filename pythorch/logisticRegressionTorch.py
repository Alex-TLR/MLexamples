''' 
Author: Aleksej Avramovic
Last update: 04/05/2024

Inspired by: Deep Learning with PyTorch Live Course (freeCodeCamp.org)
             Lecture on Logistic regression

Logistic regression implementation from the scratch.
The shuffle for train/test split is disabled. Every new run should give the same results.
'''

import torch 
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import numpy as np

dataset = MNIST(root = 'mnist/', download = False)

# print(dataset[0])
# print(len(dataset))

image, label = dataset[0]
# Uncomment to show an image
# import matplotlib.pyplot as plt
# plt.figure(figsize=(4, 4))
# plt.imshow(image, cmap = 'gray')
# plt.title(f'Label: {label}')
# plt.show()

dataset_train = MNIST(root = 'mnist/',  train = True, transform = transforms.ToTensor())
dataset_test = MNIST(root = 'mnist/', train = False, transform = transforms.ToTensor())

train_tensor, train_label = dataset_train[0]
# print(train_tensor.shape, label)

from torch.utils.data import random_split
train_data, val_data = random_split(dataset_train, [50000, 10000])
print("Train data length ", len(train_data))
print("Valid data length ", len(val_data))

from torch.utils.data import DataLoader

# Batch size
batchSize = 256

# Learning rate
lr = 1e-3

# Number of epochs
numberOfEpochs = 50

# Number of classes
numberOfClasses = 10

# Define train and validation loader with the cunks of the batchSize size
train_loader = DataLoader(train_data, batchSize, shuffle=True)
val_loader = DataLoader(val_data, batchSize)
test_loader = DataLoader(dataset_test, batchSize)

# In logistic regression, 
# model is prediction = x * w + b
# softmax is applied in order to convert output values into probabilities
# cross-entropy as a measure of loss

import torch.nn as nn
imageSize = train_tensor.size()
inputSize = imageSize[1] * imageSize[2]

import sys
sys.path.append('../')

# Utility functions

def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
    percent = f'{100 * (iter / (float(total))):.1f}'
    filled = int(length * iter // total)
    bar = fill * filled + '_' * (length - filled) + ' ' + percent
    sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
    sys.stdout.flush()

import torch.nn.functional as f 
lossFunction = f.cross_entropy

# Define model using PyTorch base class
class logisticRegressionModel(nn.Module):

    def __init__(self, iSize, nClasses):
        super().__init__()
        self.inputSize = iSize
        self.numOfClasses = nClasses
        self.model = nn.Linear(iSize, nClasses)

    def forward(self, inputData):
        inputData = inputData.reshape(-1, self.inputSize)
        outputData = self.model(inputData)
        return outputData
        # exponents = torch.exp(outputData)
        # probabilities = exponents / torch.sum(exponents)
        # maxProb, predictions = torch.max(probabilities, dim=1)
        # return predictions

    def lgSoftmax(self, input):
        e = torch.exp(input)
        p = e / torch.sum(e)
        return p
    
    def lgCrossEntropy(self, outputData, labels, nClasses):
        loss = torch.tensor(0.)
        encoded = np.zeros((len(labels)), nClasses)
        for i, l in enumerate(labels):
            encoded[i, l] = 1
        for i in range(len(labels)):
            o = self.lgSoftmax(outputData[i, :])
            o = o.detach().numpy()
            index = np.argmax(encoded[i,:])
            loss -= torch.from_numpy(np.log(o[int(index)]))
        loss /= len(labels)
        return loss
    
    def training(self, batch):
        images, labels = batch 
        output = self.model(images)
        loss = lossFunction(output, labels)
        # TODO check type of loss and check type and value of lgCrossEentropy
        return loss

        
def myOneHotEncoder(labels, numberOfClasses):
    # print("My one hot encoder")
    # print(labels)
    # print(numberOfClasses)
    encoded = np.zeros((len(labels), numberOfClasses))
    for i, l in enumerate(labels):
        encoded[i, l] = 1
    return encoded

def mySoftmax(input):
    e = torch.exp(input)
    p = e / torch.sum(e)
    return p
    # print(p.size())
    # _, predictions = torch.max(p, dim=0)
    # return predictions

def accuracy(predictions, truth):
    a = torch.tensor(torch.sum(predictions == truth).item() / len(truth))
    # print("In accuracy ", type(a))
    return a

def myCrossEntropy(output, labels, numOfClasses):
    # print("My Loss")
    loss = 0
    # print("Labels shape", labels.shape)
    labelsOHE = myOneHotEncoder(labels, numOfClasses) # encoder.fit_transform(labels.reshape(-1,1))
    # print(output)
    # print(labelsOHE)
    for i in range(len(labels)):
        # for j in range(numberOfClasses):
        # print(type(output[i,:]))
        # print(type(labelsOHE[i,:]))
        # o = output[i, :].detach().numpy()
        o = mySoftmax(output[i, :])
        o = o.detach().numpy()
        # print(type(o))
        # print(type(labelsOHE[i,:]))
        # print(f'index {i}, loss {loss}, output {o}, labelOHE {labelsOHE[i,:]} \n')
        index = np.argmax(labelsOHE[i,:])
        loss -= np.log(o[int(index)])
    loss /= len(labels)
    return loss
    # return None

    
thisModel = logisticRegressionModel(inputSize, numberOfClasses)
# print(thisModel.model.weight.shape, thisModel.model.bias.shape)
# print(list(thisModel.parameters()))  

'''
for epoch in range(numEpoch):
    for batch in train_loader:
        generate prediction
        loss
        gradients
        update weights
        reset gradients
    
    # Validate
    for batch in val_loader:
        generate predictions
        loss
        calculate metrics
    # Calculate average validation loss and metrics

    # Log epoch, loss and metrics for inspection
'''

# Define optimization
opt = torch.optim.SGD(thisModel.parameters(), lr = lr)

def fit(epochs, model, lossFunction, opt, train_data, val_data):

    for i in range(epochs):
        # Training
        tAcc = []
        tLoss = []
        for batch, labels in train_data:
            # Generate prediction
            output = model(batch)
            # Calculate loss
            # myloss = myCrossEntropy(output, labels, numberOfClasses)
            loss = lossFunction(output, labels)
            tLoss.append(loss.item())
            # Calculate gradients
            loss.backward()
            # Update weights/parameters
            opt.step()
            # Reset gradients
            opt.zero_grad()
            # Check accuracy (not necessary)
            p = model.lgSoftmax(output)
            _, o = torch.max(p, dim=1)
            a = accuracy(o, labels)
            tAcc.append(a.item())
        # Training stats
        meanTA = sum(tAcc) / len(tAcc)
        meanTL = sum(tLoss) / len(tLoss)
        
        # Validation
        vAcc = []
        vLoss = []
        for batch, labels in val_data:
            # Generate prediction
            output = model(batch)
            # Calculate loss
            l = lossFunction(output, labels)
            vLoss.append(l.item())
            # Find largest probability
            p = model.lgSoftmax(output)
            _, o = torch.max(p, dim=1)
            # Calculate metrics
            a = accuracy(o, labels)
            vAcc.append(a.item())
            # output = [model.validation_step(batch) for batch in val_data]
            # return model.validation_epoch_end(output)
        # Validation stats
        meanA = sum(vAcc) / len(vAcc)
        meanL = sum(vLoss) / len(vLoss)

        suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' + \
        'Validation loss: ' + f'{meanL:.2f} ' + 'Validation accuracy: ' + f'{meanA:.2f}'

        progressBar(i + 1, epochs, prefix = 'Progress: ', suffix = suffixArray, length = 60, fill = '#')

fit(numberOfEpochs, thisModel, lossFunction, opt, train_loader, val_loader)
print("\n")

print("Check test images.")
testAcc = []
testLoss = []
for batch, labels in test_loader:
    output = thisModel(batch)
    l = lossFunction(output, labels)
    testLoss.append(l.item())
    p = thisModel.lgSoftmax(output)
    _, o = torch.max(p, dim=1)
    a = accuracy(o, labels)
    testAcc.append(a.item())
# Validation stats
meanA = sum(testAcc) / len(testAcc)
meanL = sum(testLoss) / len(testLoss)
print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA:.2f}.')