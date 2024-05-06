''' 
Author: Aleksej Avramovic
Last update: 05/05/2024

Inspired by: Deep Learning with PyTorch Live Course (freeCodeCamp.org)
             Lecture on Feed forward networks

'''
import torch 
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
import numpy as np 
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
sys.path.append('../')

# TODO Create the graphs
# TODO GPU mechanics

# Utility functions

def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
    percent = f'{100 * (iter / (float(total))):.1f}'
    filled = int(length * iter // total)
    bar = fill * filled + '_' * (length - filled) + ' ' + percent
    sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
    sys.stdout.flush()

def accuracy(predictions, truth):
    a = torch.tensor(torch.sum(predictions == truth).item() / len(truth))
    return a

# TODO: Try with different datasets (CIFAR10)
# Visit Kaggle
# Class labels for CIFAR-10
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transformData = transforms.Compose([transforms.ToTensor()])
# transformData = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Change download= to True or False if necessary
# dataset = MNIST(root = 'mnist/', download = False)
dataset = CIFAR10(root = 'cifar10/', download = False)

# Transform data to tensor (specific to the torchvision.datasets)
# dataset_train = MNIST(root = 'mnist/',  train = True, transform = transformData)
# dataset_test = MNIST(root = 'mnist/', train = False, transform = transformData)
dataset_train = CIFAR10(root = 'cifar10/',  train = True, transform = transformData)
dataset_test = CIFAR10(root = 'cifar10/', train = False, transform = transformData)

# Get the image size
train_tensor, train_label = dataset_train[0]
imageSize = train_tensor.size()
print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')
inputSize = imageSize[0] * imageSize[1] * imageSize[2]

# Get size of the training data
dataSize = len(dataset_train)
tSize = int(0.8 * dataSize)
vSize = dataSize - tSize

from torch.utils.data import random_split
torch.manual_seed(42)
train_data, val_data = random_split(dataset_train, [tSize, vSize])
print("Train data length ", len(train_data))
print("Valid data length ", len(val_data))

# Batch size
batchSize = 32

# Learning rate (KEY)
lr = [0.2, 0.1, 0.01]

# Number of epochs
numberOfEpochs = [32, 32, 25]

# Check the LR nE consistency
assert(len(lr) == len(numberOfEpochs)), "Number of epochs must be equal to the LR"

# Number of classes
numberOfClasses = 10

# Define train and validation loader with the chunks of the batchSize size
train_loader = DataLoader(train_data, batchSize, shuffle=True)
val_loader = DataLoader(val_data, batchSize*2)
test_loader = DataLoader(dataset_test, batchSize)

# TODO: Recheck, implement RELU from the scratch
# TODO: Try with sigmoid, tanh, 
import torch.nn.functional as f 
lossFunction = f.cross_entropy
relu = f.relu

class feedForwardModel(nn.Module):

    def __init__(self, iSize, hSize, nClasses):
        super().__init__()
        self.inputSize = iSize
        self.hiddenSize = hSize
        self.numOfClasses = nClasses
        self.layer1 = nn.Linear(iSize, hSize)
        self.layer2 = nn.Linear(hSize, nClasses)
        self.activation = relu

    def forward(self, inputData):
        inputData = inputData.reshape(-1, self.inputSize)
        outputData = self.layer1(inputData)
        outputData = self.activation(outputData)
        outputData = self.layer2(outputData)
        return outputData
    
    def lgSoftmax(self, input):
        e = torch.exp(input)
        p = e / torch.sum(e)
        return p

class feedForwardModelDeep(nn.Module):
    
    def __init__(self, iSize, hSize1, hSize2, nClasses):
        super().__init__()
        self.inputSize = iSize
        self.hiddenSize1 = hSize1
        self.hiddenSize2 = hSize2
        self.numOfClasses = nClasses
        self.layer1 = nn.Linear(iSize, hSize1)
        self.layer2 = nn.Linear(hSize1, hSize2)
        self.layer3 = nn.Linear(hSize2, hSize2)
        self.layer4 = nn.Linear(hSize2, nClasses)
        self.activation = relu

    def forward(self, inputData):
        inputData = inputData.reshape(-1, self.inputSize)
        outputData = self.layer1(inputData)
        outputData = self.activation(outputData)
        outputData = self.layer2(outputData)
        outputData = self.activation(outputData)
        outputData = self.layer3(outputData)
        outputData = self.activation(outputData)
        outputData = self.layer4(outputData)
        return outputData
    
    def lgSoftmax(self, input):
        e = torch.exp(input)
        p = e / torch.sum(e)
        return p

# Define model
# thisModel = feedForwardModelDeep(inputSize, 92, 48, numberOfClasses)
thisModel = feedForwardModelDeep(inputSize, 128, 64, numberOfClasses)
# print(list(thisModel.parameters()))
print(thisModel)

# Define history
# Keeps accuracy and loss for both training and validation in each epoch
H = []

def fit(epochs, model, lossFunction, lr, train_data, val_data, history):

    opt = torch.optim.SGD(model.parameters(), lr = lr)
    for i in range(epochs):
        # Training
        tAcc = []
        tLoss = []
        for batch, labels in train_data:
            # Generate prediction
            output = model(batch)
            # Calculate loss
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
        # Validation stats
        meanA = sum(vAcc) / len(vAcc)
        meanL = sum(vLoss) / len(vLoss)

        suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' + \
        'Validation loss: ' + f'{meanL:.2f} ' + 'Validation accuracy: ' + f'{meanA:.2f}'

        progressBar(i + 1, epochs, prefix = 'Progress: ', suffix = suffixArray, length = 60, fill = '#')
        
        currentHistory = [meanTL, meanTA, meanL, meanA]
        history.append(currentHistory)

    print('\n')
    return model, history

for i in range(len(lr)):
    lrCurrent = lr[i]
    nEpochs = numberOfEpochs[i]
    thisModel, H = fit(nEpochs, thisModel, lossFunction, lrCurrent, train_loader, val_loader, H)
print("\n")

# print(H)
# Plot loss/accuracy
import matplotlib.pyplot as plt
# Extract validation loss/acc
vLoss = [v[2] for v in H]
vAcc = [v[3] for v in H]

plt.figure(figsize = (10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(len(H)), vLoss)
plt.title("Validation loss")
plt.subplot(1, 2, 2)
plt.plot(range(len(H)), vAcc)
plt.title("Validation accuracy")
plt.show()

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