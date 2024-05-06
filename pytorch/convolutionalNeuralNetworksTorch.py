''' 
Author: Aleksej Avramovic
Last update: 07/05/2024

Inspired by: Deep Learning with PyTorch Live Course (freeCodeCamp.org)
             Lecture on Image classification

'''

import torch 
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import numpy as np 
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Utility functions

def showBatch(inputData):
    for images, labels in inputData:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

# TODO: Move to BasicModel
def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
    percent = f'{100 * (iter / (float(total))):.1f}'
    filled = int(length * iter // total)
    bar = fill * filled + '_' * (length - filled) + ' ' + percent
    sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
    sys.stdout.flush()

# TODO: Move to BasicModel
def accuracy(predictions, truth):
    a = torch.tensor(torch.sum(predictions == truth).item() / len(truth))
    return a

# Import CIFAR10 data
transformData = transforms.Compose([transforms.ToTensor()])
dataset = CIFAR10(root = 'cifar10/', download = False)
dataset_train = CIFAR10(root = 'cifar10/',  train = True, transform = transformData)
dataset_test = CIFAR10(root = 'cifar10/', train = False, transform = transformData)

# Print classe
print(dataset.classes)

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
gen1 = torch.Generator().manual_seed(42)
train_data, val_data = random_split(dataset_train, [tSize, vSize], generator=gen1)
print("Train data length ", len(train_data))
print("Valid data length ", len(val_data))

# Batch size
batchSize = 128

# Learning rate (KEY)
lr = [0.1, 0.01]

# Number of epochs
numberOfEpochs = [25, 5]

# Check the LR nE consistency
assert(len(lr) == len(numberOfEpochs)), "Number of epochs must be equal to the LR"

# Number of classes
numberOfClasses = 10

# Define train and validation loader with the chunks of the batchSize size
train_loader = DataLoader(train_data, batchSize, shuffle=True)
val_loader = DataLoader(val_data, batchSize*2)
test_loader = DataLoader(dataset_test, batchSize)

import torch.nn.functional as f 

# showBatch(train_loader)

class BasicModel(nn.Module):

    def __init__(self, nClasses):
        super().__init__()
        self.numberOfClasses = nClasses

    # TODO: Make test for this
    # TODO: Extend this function for multiple inputs
    def softmax(self, input):
        e = torch.exp(input)
        p = e / torch.sum(e)
        return p 
    
    def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
        percent = f'{100 * (iter / (float(total))):.1f}'
        filled = int(length * iter // total)
        bar = fill * filled + '_' * (length - filled) + ' ' + percent
        sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
        sys.stdout.flush()
        return None
    
    def accuracy(self, pred, truth):
        _, o = torch.max(pred, dim = 1)
        return torch.tensor(torch.sum(o == truth).item() / len(truth))
    
    def fit(self, nEpochs, model, lossFunction, lr,  train_load, val_load, history):
        '''
        nEpochs:      number of epochs for training
        model:        architecure of network
        lossFunction: loss function
        lr:           learning rate
        train_load:   loader for train data
        val_load:     loader for validation data
        history:      history of the statistics of previous batches
        '''

        # Define optimization
        opt = torch.optim.SGD(model.parameters(), lr = lr)
        for i in range(nEpochs):
            # Training
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                # Generate predictions
                pred = model(batch)
                # Calculate loss
                loss = lossFunction(pred, labels)
                tLoss.append(loss.item())
                # Calculate gradients
                loss.backward()
                # Update parameters
                opt.step()
                # Reset gradiensts
                opt.zero_grad()
                # Check train accuracy
                a = self.accuracy(pred, labels)
                tAcc.append(a.item())
            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)

            # Validation
            # Define lists to store validation loss and accuracy
            vLoss = []
            vAcc = list()
            for batch, labels in val_load:
                pred = model(batch)
                loss = lossFunction(pred, labels)
                vLoss.append(loss.item())
                a = self.accuracy(pred, labels)
                vAcc.append(a.item())
            # Validation stats
            meanVA = sum(vAcc) / len(vAcc)
            meanVL = sum(vLoss) / len(vLoss) 

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' + \
            'Validation loss: ' + f'{meanVL:.2f} ' + 'Validation accuracy: ' + f'{meanVA:.2f}'

            progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 60, fill = '#')
            currentHistory = [meanTL, meanTA, meanVL, meanVA]
            history.append(currentHistory)
        
        print('\n')
        return model, history
   
    
class Cifar10Model(BasicModel):

    def __init__(self, nClasses):
        super().__init__(nClasses)
        self.network = nn.Sequential()

        self.network.add_module('conv1_1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
        self.network.add_module('relu1_1', nn.ReLU())
        self.network.add_module('conv1_2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        self.network.add_module('relu1_2', nn.ReLU())
        self.network.add_module('maxp1_2', nn.MaxPool2d(2,2))

        self.network.add_module('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.network.add_module('relu2_1', nn.ReLU())
        self.network.add_module('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.network.add_module('relu2_2', nn.ReLU())
        self.network.add_module('maxp2_2', nn.MaxPool2d(2,2))

        self.network.add_module('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.network.add_module('relu3_1', nn.ReLU())
        self.network.add_module('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.network.add_module('relu3_2', nn.ReLU())
        self.network.add_module('maxp3_2', nn.MaxPool2d(2,2))

        self.network.add_module('flat4_1', nn.Flatten())
        self.network.add_module('dens4_1', nn.Linear(4096, 1024))
        self.network.add_module('relu4_1', nn.ReLU())
        self.network.add_module('dens4_2', nn.Linear(1024, 512))
        self.network.add_module('relu4_2', nn.ReLU())
        self.network.add_module('dens4_3', nn.Linear(512, self.numberOfClasses))

    def forward(self, inputData):
        return self.network(inputData)
    
thisModel = Cifar10Model(10)
print(thisModel)

# Define loss function
import torch.nn.functional as f 
lossFunction = f.cross_entropy

# Define history
# Keeps accuracy and loss for both training and validation in each epoch
H = []

for i in range(len(lr)):
    lrCurrent = lr[i]
    nEpochs = numberOfEpochs[i]
    thisModel, H = thisModel.fit(nEpochs, thisModel, lossFunction, lrCurrent, train_loader, val_loader, H)
print("\n")

import matplotlib.pyplot as plt
# Extract validation loss/acc
tLoss = [v[0] for v in H]
vLoss = [v[2] for v in H]

plt.figure(figsize = (5, 5))
plt.plot(tLoss, '-bx')
plt.plot(vLoss, '-rx')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(['Training', 'Validation'])
plt.title('Loss/epochs')
plt.show()

print("Check test images.")
testAcc = []
testLoss = []
for batch, labels in test_loader:
    output = thisModel(batch)
    l = lossFunction(output, labels)
    testLoss.append(l.item())
    # p = thisModel.lgSoftmax(output)
    _, o = torch.max(output, dim=1)
    a = accuracy(o, labels)
    testAcc.append(a.item())
# Validation stats
meanA = sum(testAcc) / len(testAcc)
meanL = sum(testLoss) / len(testLoss)
print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA:.2f}.')