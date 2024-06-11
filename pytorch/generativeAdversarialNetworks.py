
'''
https://www.youtube.com/watch?v=ABaZ_tecZ3U&list=PLWKjhJtqVAbm3T2Eq1_KgloC7ogdXxdRa&index=10
'''
import torch 

from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms
import numpy as np 
from torch.utils.data import DataLoader
import torch.nn as nn
import sys
sys.path.append('../')
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt

def imageDenorm(imageTensor):
    return imageTensor * 0.5 + 0.5

def showBatch(inputData):
    # for images in inputData:
    fig, ax = plt.subplots(figsize=(16,16))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imageDenorm(inputData.detach()[:64]), nrow=8).permute(1, 2, 0))
    plt.show()
        # break

# def showBatch2(inputData):
#     for images in inputData:
#         fig, ax = plt.subplots(figsize=(8,8))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.imshow(make_grid(imageDenorm(images.detach()), nrow=8).permute(1, 2, 0))
#         plt.show()
def showBatch2(inputData):
    fig, ax = plt.subplots(figsize=(16, 16))
    ax.set_xticks([])
    ax.set_yticks([])
    images_grid = make_grid(imageDenorm(inputData.detach()), nrow=8).permute(1, 2, 0)
    ax.imshow(images_grid)
    plt.show()

def progressBar(iter, total, prefix = '', suffix = '', length = 30, fill = '#'):
    percent = f'{100 * (iter / (float(total))):.1f}'
    filled = int(length * iter // total)
    bar = fill * filled + '_' * (length - filled) + ' ' + percent
    sys.stdout.write('\r%s |%s%% %s' % (prefix, bar, suffix))
    sys.stdout.flush()

def accuracy(predictions, truth):
    a = torch.tensor(torch.sum(predictions == truth).item() / len(truth))
    return a

transformData = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5), (0.5))])
# Above normalization takes images with the pixel range from 0 to 1 and converts to -1 to 1

dataset = MNIST(root = 'mnist/', download = False)

dataset_train = MNIST(root = 'mnist/',  train = True, transform = transformData)
dataset_test = MNIST(root = 'mnist/', train = False, transform = transformData)

# Get the image size
train_tensor, train_label = dataset_train[0]
imageSize = train_tensor.size()
print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')
inputSize = imageSize[0] * imageSize[1] * imageSize[2]

# print(train_tensor)

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
batchSize = 64

# Latent size
latentSize = 64

import os
sampleDirectory = 'generatedImages'
os.makedirs(sampleDirectory, exist_ok=True)

# Define train and validation loader with the chunks of the batchSize size
train_loader = DataLoader(train_data, batchSize, shuffle=True)

# showBatch(train_loader)

# Define loss function
import torch.nn.functional as f 
lossFunction = f.binary_cross_entropy

# fixed_latent
fixedLatent = torch.randn(batchSize, latentSize, 1, 1)

class BasicGANModel(nn.Module):

    def __init__(self, numOfChannels, latentSize):
        super().__init__()
        self.numOfChannels = numOfChannels
        self.latentSize = latentSize
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator()
    
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
        # Forward pass through the generator
        return self.generator(x)

    def define_discriminator(self):
        pass

    def forward_discriminator(self):
        pass
    
    def define_generator(self):
        pass

    def forward_generator(self):
        pass
    
    def saveFakeImages(self, index, latentTensor, show=False):
        fake_images = self.generator(latentTensor)
        fake_iNames = 'generated_images_{0:0=4d}.png'.format(index)
        save_image(imageDenorm(fake_images), os.path.join(sampleDirectory, fake_iNames))
        # print("Saving, ", fake_iNames)
        if show:
            showBatch(fake_images)

    def trainDiscriminator(self, images, optD):
        optD.zero_grad()

        # Pass real images through discriminator
        realPred = self.forward_discriminator(images)
        # Generate labels fo real images
        # All real images must have label equal to one
        realLabels = torch.ones(images.size(0), 1)
        realLoss = lossFunction(realPred, realLabels)
        realScore = torch.mean(realPred).item()

        # Generate fake images
        latent = torch.randn(batchSize, latentSize, 1, 1)
        fakeImages = self.generator(latent)

        # Pass fake images through discriminator
        # Set all fake labels to zero
        fakeLabels = torch.zeros(images.size(0), 1)
        fakePred = self.forward_discriminator(fakeImages)
        fakeLoss = lossFunction(fakePred, fakeLabels)
        fakeScore = torch.mean(fakePred).item()

        # Update loss
        loss = realLoss + fakeLoss
        # Calculate gradients
        loss.backward()
        # Update network parameters
        optD.step()

        return loss.item(), realScore, fakeScore
    
    def trainGenerator(self, optG):
        optG.zero_grad()

        # Generate fake images
        latent = torch.randn(batchSize, latentSize, 1, 1)
        fakeImages = self.forward_generator(latent)

        # Fool discriminator
        # In this case, every fake image has label 1 (like real image), so the loss will be calculated according the case
        # that fake image is equal to real image
        fakeLabels = torch.ones(fakeImages.size(0), 1)
        fakePred = self.forward_discriminator(fakeImages)
        fakeLoss = lossFunction(fakePred, fakeLabels)

        # Update generator
        fakeLoss.backward()
        optG.step()

        return fakeLoss.item()
    
    # Needs revision
    def fit(self, nEpochs, lr, start_index=1):
    
        '''
        nEpochs:      number of epochs for training
        lr:           learning rate
        '''
        # Define optimization
        optDiscriminator = torch.optim.Adam(self.discriminator.parameters(), lr = lr, betas=(0.5, 0.999))
        optGenerator = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.999))

        # Training
        for i in range(nEpochs):

            # Losses and scores
            lossesGenerator = []
            lossesDiscriminator = []
            scoresRealImages = []
            scoresFakeImages = []
            # Training
            for batch, labels in train_loader:
                lossD, realScore, fakeScore = self.trainDiscriminator(batch, optDiscriminator)
                lossG = self.trainGenerator(optGenerator)

                lossesDiscriminator.append(lossD)
                lossesGenerator.append(lossG)
                scoresRealImages.append(realScore)
                scoresFakeImages.append(fakeScore)

            # Training stats
            meanSR = sum(scoresRealImages) / len(scoresRealImages)
            meanSF = sum(scoresFakeImages) / len(scoresFakeImages)

            meanLD = sum(lossesDiscriminator) / len(lossesDiscriminator)
            meanLG = sum(lossesGenerator) / len(lossesGenerator) 

            # Make progress bar
            suffixArray = ' ' + 'Real image score: ' + f'{meanSR:.2f} ' + 'Fake image score: ' + f'{meanSF:.2f} ' + \
            'Discriminator loss: ' + f'{meanLD:.2f} ' + 'Generator loss: ' + f'{meanLG:.2f}'

            self.progressBar(i + 1, nEpochs, prefix = 'Progress: ', suffix = suffixArray, length = 40, fill = '#')
            # currentHistory = [meanTL, meanTA, meanVL, meanVA]
            # history.append(currentHistory)
            self.saveFakeImages(i + start_index, fixedLatent)
        
        print('\n')
        return None

class MyGANModelMNIST(BasicGANModel):

    def __init__(self, numOfChannels, latentSize):
        super().__init__(numOfChannels, latentSize)
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator()

    def define_discriminator(self):
                 # 1 x 28 x 28
        layers = [nn.Conv2d(self.numOfChannels, 64, kernel_size=2, stride=2, padding=0, bias=False),
                  nn.BatchNorm2d(64),
                  nn.LeakyReLU(0.2, inplace=True),
                 # 64 x 14 x 14
                  nn.Conv2d(64, 128, kernel_size=2, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(128),
                  nn.LeakyReLU(0.2, inplace=True), 
                 # 128 x 8 x 8
                  nn.Conv2d(128, 256, kernel_size=2, stride=2, padding=0, bias=False),
                  nn.BatchNorm2d(256),
                  nn.LeakyReLU(0.2, inplace=True),
                 # 256 x 4 x 4
                  nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0, bias=False),
                 # 1 x 1 x 1
                  nn.Flatten(),
                  nn.Sigmoid()]
        return nn.Sequential(*layers)
        # return network(inputData)

    def forward_discriminator(self, x):
        return self.discriminator(x)
    
    def define_generator(self):
                 # latentSize x 1 x 1
        layers = [nn.ConvTranspose2d(self.latentSize, 256, kernel_size=4, stride=1, padding=0, bias=False),
                  nn.BatchNorm2d(256),
                  nn.ReLU(True),
                 # 256 x 4 x 4
                  nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=False),
                  nn.BatchNorm2d(128),
                  nn.ReLU(True),
                 # 128 x 8 x 8
                  nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=1, bias=False),
                  nn.BatchNorm2d(64),
                  nn.ReLU(True),
                 # 64 x 14 x 14
                  nn.ConvTranspose2d(64, self.numOfChannels, kernel_size=2, stride=2, padding=0, bias=False),
                  nn.Tanh()]
        return nn.Sequential(*layers)

    def forward_generator(self, x):
        return self.generator(x)

class SimpleGANModelMNIST(BasicGANModel):
    def __init__(self, numOfChannels, latentSize):
        super().__init__(numOfChannels, latentSize)
        self.discriminator = self.define_discriminator()
        self.generator = self.define_generator()

    def define_discriminator(self):
                # 1 x 28 x 28
        layers = [nn.Conv2d(self.numOfChannels, 64, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                # 64 x 14 x 14
                nn.Conv2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True), 
                # 64 x 7 x 7
                nn.Conv2d(64, 1, kernel_size=7, stride=1, padding=0, bias=False),
                # 1 x 1 x 1
                nn.Flatten(),
                nn.Sigmoid()]
        return nn.Sequential(*layers)
    
    def forward_discriminator(self, x):
        return self.discriminator(x)
    
    def define_generator(self):
                # latentSize x 1 x 1
        layers = [nn.ConvTranspose2d(self.latentSize, 64, kernel_size=7, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # 64 x 7 x 7
                nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # 64 x 14 x 14
                nn.ConvTranspose2d(64, self.numOfChannels, kernel_size=2, stride=2, padding=0, bias=False),
                nn.Tanh()]
        return nn.Sequential(*layers)

    def forward_generator(self, x):
        return self.generator(x)

    
thisModel = SimpleGANModelMNIST(1, latentSize)
xb = torch.randn(batchSize, latentSize, 1, 1)
print(xb.shape)
fake_images = thisModel.generator(xb)
print(fake_images.shape)
showBatch2(fake_images)

learningRate = 0.0002
numOfE = 50

thisModel.fit(numOfE, learningRate, 1)