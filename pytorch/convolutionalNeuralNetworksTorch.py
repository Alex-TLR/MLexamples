''' 
Author: Aleksej Avramovic
Last update: 07/05/2024

Inspired by: Deep Learning with PyTorch Live Course (freeCodeCamp.org)
             Lectures on Image classification, ResNet, Regularization and Data augmentation

             https://www.youtube.com/watch?v=TN9fMYQxw4E&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa&index=4
             https://www.youtube.com/watch?v=sJF6PiAjE1M&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa&index=5
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
torch.autograd.set_detect_anomaly(True)
# Utility functions

def showBatch(inputData):
    for images, labels in inputData:
        fig, ax = plt.subplots(figsize=(12,6))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

# Import CIFAR10 data
# transformData = transforms.Compose([transforms.ToTensor()])
transformData = transforms.Compose([
    transforms.RandomCrop(32, 4, padding_mode='reflect'),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.492, 0.483, 0.447), std=(0.55, 0.54, 0.518))])
# mean = (0.4914, 0.4822, 0.4464)
# std = (0.2023, 0.1994, 0.2010)

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
tSize = int(0.9 * dataSize)
vSize = dataSize - tSize

from torch.utils.data import random_split
gen1 = torch.Generator().manual_seed(42)
train_data, val_data = random_split(dataset_train, [tSize, vSize], generator=gen1)
print("Train data length ", len(train_data))
print("Valid data length ", len(val_data))

# TODO: why larger size does not work properly
# Batch size
batchSize = 256

# Learning rate (KEY)
lr = [0.01]

# Weight decay
wDecay = 0.0001

# Gradient clipping
gClip = 0.1

# Number of epochs
numberOfEpochs = [20]

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
    
    def progressBar(self, iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
        percent = f'{100 * (iter / (float(total))):.1f}'
        filled = int(length * iter // total)
        bar = fill * filled + '_' * (length - filled) + ' ' + percent
        sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
        sys.stdout.flush()
        return None
    
    def accuracy(self, pred, truth):
        _, o = torch.max(pred, dim = 1)
        return torch.tensor(torch.sum(o == truth).item() / len(truth))
    
    # TODO: Define train and valid steps separately 
    # also validation needs to have decorator @torch.no_grad()
    
    def fit(self, nEpochs, model, lossFunction, lr, train_load, val_load, history, wd = 0, gd = None):
        '''
        nEpochs:      number of epochs for training
        model:        architecure of network
        lossFunction: loss function
        lr:           learning rate
        wd:           weight decay
        gd:           gradient clipping
        train_load:   loader for train data
        val_load:     loader for validation data
        history:      history of the statistics of previous batches
        '''

        # Define optimization
        opt = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, epochs=nEpochs, steps_per_epoch=len(train_load))
        for i in range(nEpochs):
            # Training
            model.train()
            # Define lists to store training loss and accuracy
            tLoss = []
            tAcc = list()
            for batch, labels in train_load:
                # Generate predictions
                pred = model(batch)
                # Calculate loss
                loss = lossFunction(pred, labels)
                tLoss.append(loss.detach().item())
                # Calculate gradients
                loss.backward()
                if gd:
                    nn.utils.clip_grad_value_(model.parameters(), gd)
                # Update parameters
                opt.step()
                sched.step()
                # Reset gradiensts
                opt.zero_grad()
                # Check train accuracy
                a = self.accuracy(pred, labels)
                tAcc.append(a.item())

                del batch, labels, pred, loss
            # Training stats
            meanTA = sum(tAcc) / len(tAcc)
            meanTL = sum(tLoss) / len(tLoss)

            # Validation
            # Define lists to store validation loss and accuracy
            vLoss = []
            vAcc = list()
            model.eval()
            for batch, labels in val_load:
                pred = model(batch)
                with torch.no_grad():
                    loss = lossFunction(pred, labels)
                vLoss.append(loss.detach().item())
                a = self.accuracy(pred, labels)
                vAcc.append(a.item())
            # Validation stats
            meanVA = sum(vAcc) / len(vAcc)
            meanVL = sum(vLoss) / len(vLoss) 

            # Make progress bar
            suffixArray = ' ' + 'Training loss: ' + f'{meanTL:.2f} ' + 'Training accuracy: ' + f'{meanTA:.2f} ' + \
            'Validation loss: ' + f'{meanVL:.2f} ' + 'Validation accuracy: ' + f'{meanVA:.2f}'

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            currentHistory = [meanTL, meanTA, meanVL, meanVA]
            history.append(currentHistory)
        
        print('\n')
        return model, history
   
    
class Cifar10Model(BasicModel):

    def __init__(self, nClasses):
        super().__init__(nClasses)
        self.network = nn.Sequential()

        self.network.add_module('conv1_1', nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1))
        self.network.add_module('bnor1_1', nn.BatchNorm2d(num_features=32))
        self.network.add_module('relu1_1', nn.ReLU())
        self.network.add_module('conv1_2', nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1))
        self.network.add_module('bnor1_2', nn.BatchNorm2d(num_features=64))
        self.network.add_module('relu1_2', nn.ReLU())
        self.network.add_module('maxp1_2', nn.MaxPool2d(2,2))

        self.network.add_module('conv2_1', nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1))
        self.network.add_module('bnor2_1', nn.BatchNorm2d(num_features=128))
        self.network.add_module('relu2_1', nn.ReLU())
        self.network.add_module('conv2_2', nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1))
        self.network.add_module('bnor2_2', nn.BatchNorm2d(num_features=128))
        self.network.add_module('relu2_2', nn.ReLU())
        self.network.add_module('maxp2_2', nn.MaxPool2d(2,2))

        self.network.add_module('conv3_1', nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1))
        self.network.add_module('bnor3_1', nn.BatchNorm2d(num_features=256))
        self.network.add_module('relu3_1', nn.ReLU())
        self.network.add_module('conv3_2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1))
        self.network.add_module('bnor3_2', nn.BatchNorm2d(num_features=256))
        self.network.add_module('relu3_2', nn.ReLU())
        self.network.add_module('maxp3_2', nn.MaxPool2d(2,2))

        self.network.add_module('flat4_1', nn.Flatten())
        self.network.add_module('dens4_1', nn.Linear(4096, 1024))
        self.network.add_module('drop4_1', nn.Dropout(0.4))
        self.network.add_module('relu4_1', nn.ReLU())
        self.network.add_module('dens4_2', nn.Linear(1024, 512))
        self.network.add_module('drop4_2', nn.Dropout(0.4))
        self.network.add_module('relu4_2', nn.ReLU())
        self.network.add_module('dens4_3', nn.Linear(512, self.numberOfClasses))

    def forward(self, inputData):
        return self.network(inputData)


# class ResidualBlock(nn.Module):

#     def __init__(self, numOfInputs, numOfOutputs, i, stride=1):
#         super(ResidualBlock, self).__init__()
#         self.block = nn.Sequential(
#             nn.Conv2d(in_channels=numOfInputs, out_channels=numOfOutputs, kernel_size=3, padding=1, stride=stride),
#             nn.BatchNorm2d(num_features=numOfOutputs),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=numOfOutputs, out_channels=numOfOutputs, kernel_size=3, padding=1, stride=stride),
#             nn.BatchNorm2d(num_features=numOfOutputs),
#             nn.ReLU()
#         )
#         # self.block = nn.Sequential()
#         # self.block.add_module('conv' + str(i) + '_1', nn.Conv2d(in_channels=numOfInputs, out_channels=numOfOutputs, kernel_size=3, padding=1, stride=stride))
#         # self.block.add_module('bnor' + str(i) + '_1', nn.BatchNorm2d(num_features=numOfOutputs))
#         # self.block.add_module('relu' + str(i) + '_1', nn.ReLU())
        
#         # self.block.add_module('conv' + str(i) + '_2', nn.Conv2d(in_channels=numOfInputs, out_channels=numOfOutputs, kernel_size=3, padding=1, stride=stride))
#         # self.block.add_module('bnor' + str(i) + '_2', nn.BatchNorm2d(num_features=numOfOutputs))
#         # self.block.add_module('relu' + str(i) + '_2', nn.ReLU())
#         # return None 
    
#     def forward(self, inputData):
#         outputData = self.block(inputData)
#         outputData += inputData  # here numOfInputs and numOfOutpus must be the same
#         print(outputData.shape)
#         return outputData
# class Cifar10ResModel(BasicModel):

#     '''
#     ResNet9 model
#     '''
#     def __init__(self, nClasses):
#         super().__init__(nClasses)
#         self.network = nn.Sequential()
#         # self.network += self.convBlock1(nClasses, 64, 1)
#         # self.network += self.convBlock2(64, 128, 2)
#         # self.network += ResidualBlock(128, 128, 3)
#         # self.network += self.convBlock2(128, 256, 4)
#         # self.network += self.convBlock2(256, 512, 5)
#         # self.network += ResidualBlock(512, 512, 6)
#         # self.network.add_module('maxp7_1', nn.MaxPool2d(4,4))
#         # self.network.add_module('flat7_1', nn.Flatten())
#         # self.network.add_module('dens7_1', nn.Linear(512, self.numberOfClasses))
#         # Add layers using self.add_block() method
#         self.add_block(self.convBlock1(3, 64, 1))
#         self.add_block(self.convBlock2(64, 128, 2))
#         self.add_block(ResidualBlock(128, 128, 3))
#         self.add_block(self.convBlock2(128, 256, 4))
#         self.add_block(self.convBlock2(256, 512, 5))
#         self.add_block(ResidualBlock(512, 512, 6))

#         self.network.add_module('maxp7_1', nn.MaxPool2d(4))
#         self.network.add_module('flat7_1', nn.Flatten())
#         self.network.add_module('dens7_1', nn.Linear(512, self.numberOfClasses))

#     def add_block(self, block):
#         self.network.add_module('block_' + str(len(self.network) + 1), block)

#     def convBlock1(self, input, output, i, stride=1):
#         block = nn.Sequential()
#         block.add_module('conv' + str(i) + '_1', nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1, stride=stride))
#         block.add_module('bnor' + str(i) + '_1', nn.BatchNorm2d(num_features=output))
#         block.add_module('relu' + str(i) + '_1', nn.ReLU())
#         return block

#     def convBlock2(self, input, output, i, stride=1):
#         block = nn.Sequential()
#         block.add_module('conv' + str(i) + '_1', nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1, stride=stride))
#         block.add_module('bnor' + str(i) + '_1', nn.BatchNorm2d(num_features=output))
#         block.add_module('relu' + str(i) + '_1', nn.ReLU())
#         block.add_module('maxp' + str(i) + '_1', nn.MaxPool2d(2,2))
#         return block

#     def forward(self, inputData):
#        return self.network(inputData)

class Cifar10Res9Model(BasicModel):

    def __init__(self, numberOfChannels, numberOfClasses):
        super().__init__(numberOfClasses)

        self.block1 = self.convBlock1(numberOfChannels, 64)
        self.block2 = self.convBlock2(64, 128)
        self.resBlock1 = nn.Sequential(self.convBlock1(128, 128), self.convBlock1(128, 128))

        self.block3 = self.convBlock2(128, 256)
        self.block4 = self.convBlock2(256, 512)
        self.resBlock2 = nn.Sequential(self.convBlock1(512, 512), self.convBlock1(512, 512))

        self.classifier = self.flatLayer(numberOfClasses)

    def convBlock1(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_features=output),
                  nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def convBlock2(self, input, output):
        layers = [nn.Conv2d(in_channels=input, out_channels=output, kernel_size=3, padding=1),
                  nn.BatchNorm2d(num_features=output),
                  nn.ReLU(inplace=True),
                  nn.MaxPool2d(2)]
        return nn.Sequential(*layers)
    
    def flatLayer(self, numOfClasses):
        layers = [nn.MaxPool2d(4),
                  nn.Flatten(),
                  nn.Linear(512, numOfClasses)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.resBlock1(out) + out 
        out = self.block3(out)
        out = self.block4(out)
        out = self.resBlock2(out) + out
        out = self.classifier(out)
        return out

thisModel = Cifar10Res9Model(3,10)
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
    thisModel, H = thisModel.fit(nEpochs, thisModel, lossFunction, lrCurrent, train_loader, val_loader, H, wd=wDecay, gd=gClip)
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
    a = thisModel.accuracy(output, labels)
    testAcc.append(a.item())
# Test stats
meanA = sum(testAcc) / len(testAcc)
meanL = sum(testLoss) / len(testLoss)
print(f'Test loss is {meanL:.2f}. Test accuracy is {meanA:.2f}.')