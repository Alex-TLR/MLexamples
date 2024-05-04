import torch 
from torchvision.datasets import MNIST
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

# TODO: Try with different datasets
# Change download= to True or False if necessary
dataset = MNIST(root = 'mnist/', download = False)

# Transforma data to tensor (specific to the torchvision.datasets)
dataset_train = MNIST(root = 'mnist/',  train = True, transform = transforms.ToTensor())
dataset_test = MNIST(root = 'mnist/', train = False, transform = transforms.ToTensor())

# Get image size
train_tensor, train_label = dataset_train[0]
imageSize = train_tensor.size()
inputSize = imageSize[1] * imageSize[2]

from torch.utils.data import random_split
train_data, val_data = random_split(dataset_train, [50000, 10000])
print("Train data length ", len(train_data))
print("Valid data length ", len(val_data))

# Batch size
batchSize = 64

# Learning rate (KEY)
lr = 0.5

# Number of epochs
numberOfEpochs = 5

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


# Define model
thisModel = feedForwardModel(inputSize, 32, numberOfClasses)
# print(thisModel.model.weight.shape, thisModel.model.bias.shape)
# print(list(thisModel.parameters()))
print(thisModel)

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