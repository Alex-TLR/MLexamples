import torch 
import snntorch as snn
from snntorch import utils
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from snntorch import spikegen

# Animation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
from IPython.display import HTML

'''
Different models for generation of 'spiking' data, based on MNIST data set.

'''

# Data must be transfered to tensors (in order to use pytorch)
# Also, tensors are normalized image-wise
transformData = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.0,), (1.0,))])

# Change download= to True or False if necessary
dataset = MNIST(root = 'mnist/', download = False)
dataset_train = MNIST(root = 'mnist/',  train = True, transform = transformData)
dataset_test = MNIST(root = 'mnist/', train = False, transform = transformData)

# Get the image size
train_tensor, train_label = dataset_train[0]
imageSize = train_tensor.size()
print(f'Image size: {imageSize[0]}, {imageSize[1]}, {imageSize[2]}')
inputSize = imageSize[0] * imageSize[1] * imageSize[2]

# Batch size
batchSize = 128

# Number of classes
numberOfClasses = 10

# Reduce number of training data
subsetValue = 10
dataset_train = utils.data_subset(dataset_train, subsetValue)
dataset_test = utils.data_subset(dataset_test, subsetValue)

print(f'Size of the training data is {len(dataset_train)}.')

# Create data loader
train_loader = DataLoader(dataset_train, batchSize, shuffle=True)


### Spike rate encoding ###

'''
Spiking networks are made to exploit time-varying data.
MNIST is not time-varying data.

1. Pass the training sample repeatedly, at each time step.
2. Convert input to a spike train:
   time series of the image-size tensors are made
   at each time step, the pixel value is considered as a probability to fire a spike at the corresponding pixel position
   For example: if pixel value is X[i,j] = 0.6, and we make 100 temporal steps, in each step there is a 0.6 probability
   to fire a spike at the position of (i,j).
'''

# Temporal steps
numberOfSteps = 100

def makeVideo(data, label, name):
    fig, ax = plt.subplots()
    anim = splt.animator(data, fig, ax)
    HTML(anim.to_html5_video())
    anim.save(name)
    print(f'The corresponding label is: {label}')

def spikeRateEncodingExample(steps, prob):
    print(f'Create a vector filled with values equal to given probability...')
    exampleVector = torch.ones(steps)*prob
    print(f'Pass example vector through the Bernulli trail...')
    rateEncodedVector = torch.bernoulli(exampleVector)
    print(f'Example vector     \n: {exampleVector}')
    print(f'Rate encoded vector\n: {rateEncodedVector}')
    print(f'The output is spiking {rateEncodedVector.sum()*100/len(rateEncodedVector):.2f}% of the time.')

def spikeRateEncodingMNIST(train_loader=train_loader, steps=numberOfSteps, showPlot=False):
    # Iterate through minibatches
    data = iter(train_loader)
    data_it, label_it = next(data)
    # Variables data_it and label_it contain the first batch (no loop at the moment)
    spike_data = spikegen.rate(data_it, num_steps=steps)
    print(f'Number of elements given by spikegen.rate is {spike_data.size()}.')
    print(f'Create an animation of spike data for the first element in the batch...')
    spikeDataSample = spike_data[:, 0, 0]
    # Considering that the size of spike_data is: [steps, batch, 1, 28, 28] for MNIST
    # spike data sample must slice each step, the first image from the batch and the first (and only for MNIST) channel 
    print(f'Number of elements after slicing is {spikeDataSample.size()}.')
    print(f'If we want different image sample from the batch, we must replace the first 0 in spike_data[:, 0, 0].')
    print(f'The first spiking video, gain=1.0')
    makeVideo(data=spikeDataSample, label=label_it[0], name="spikeRateMNIST_10.mp4")
    print(f'The second spiking video, gain=0.3')
    spike_data_2 = spikegen.rate(data_it, num_steps=steps, gain=0.3)
    print(f'Number of elements given by spikegen.rate is {spike_data_2.size()}.')
    spikeDataSample2 = spike_data_2[:, 0, 0]
    makeVideo(data=spikeDataSample2, label=label_it[0], name="spikeRateMNIST_03.mp4")

    if (showPlot):
        print(f'Make pics...')
        plt.close('all')
        plt.switch_backend('TkAgg')
        plt.figure(facecolor="w")
        plt.subplot(1,2,1)
        plt.imshow(spikeDataSample.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
        plt.axis('off')
        plt.title('Gain = 1')
        plt.subplot(1,2,2)
        plt.imshow(spikeDataSample2.mean(axis=0).reshape((28,-1)).cpu(), cmap='binary')
        plt.axis('off')
        plt.title('Gain = 0.3')
        plt.show()

    return None


# TODO Recheck 
# # Raster plot of an input sample???
# # We need to reshape the sample into a 2D tensor, where  time is the first dimension# I am not sure if this is necessary

# spikeDataSample3 = spike_data_2.reshape((numberOfSteps, -1))
# print(spikeDataSample3.size())
# fig = plt.figure(facecolor="w", figsize=(10, 5))
# ax = fig.add_subplot(111)
# splt.raster(spikeDataSample3, ax, s=1.5, c="black")

# plt.title("Input Layer")
# plt.xlabel("Time step")
# plt.ylabel("Neuron Number")
# plt.show()


### Latency coding ###
'''
Spike is emitted when voltage V(t) reaches a threshold value Vthr.
This model is based on the parallel connection of resistor (R) and capacitor (C). The RC is charged with current Iin(t).
When voltage V(t) over RC connection reaches the Vthr, the spike is emitted.

Time nedded for spike to be emotted, in the case when constant current is used to charge RC, is calculared with the following:
t = RC * ln(R*Iin / (R*Iin - Vthr)).

In this case, each input neuron can fire only one during the time span.
'''

def convert_to_time(data, tau=5, threshold=0.01):
    '''
    In this simplified model, R = 1, so C = tau
    '''
    spike_time = tau * torch.log(data / (data - threshold))
    return spike_time

def spikeLatencyEncodingExample():
    print(f'Make a test tensor')
    raw_input = torch.arange(0, 5, 0.05) # tensor from 0 to 5
    print(raw_input)
    spike_times = convert_to_time(raw_input)
    print(spike_times)
    plt.close('all')
    plt.switch_backend('TkAgg')
    plt.plot(raw_input, spike_times)
    plt.xlabel('Input Value')
    plt.ylabel('Spike Time (s)')
    plt.show()

# def spikeLatencyEncodingMNIST(train_loader=train_loader, steps=numberOfSteps, showPlot=False):
def spikeLatencyEncodingMNIST(train_loader=train_loader, steps=numberOfSteps):
    data = iter(train_loader)
    data_it, label_it = next(data)
    spike_data = spikegen.latency(data_it, steps, tau=5, threshold=0.01)
    print(f'Lenght {len(spike_data)}.')

    spikeDataSample = spike_data[:, 0, 0]
    makeVideo(data=spikeDataSample, label=label_it[0], name="spikeLatencyMNIST_1.mp4")

    plt.close('all')
    plt.switch_backend('TkAgg')
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    x = spike_data[:, 0].view(steps, -1)
    print(x.size())
    splt.raster(spike_data[:, 0].view(steps, -1), ax, s=25, c="black")
    # Set x-axis tick locations and labels
    ax.set_xlim(-10, len(spike_data)) 
    ax.set_xticks(range(0, len(spike_data)+1, 10))
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

    spike_data_2 = spikegen.latency(data_it,steps, tau=5, threshold=0.01, linear=True, clip=True, normalize=True)
    print(f'Lenght {len(spike_data_2)}.')
    spikeDataSample2 = spike_data_2[:, 0, 0]
    makeVideo(data=spikeDataSample2, label=label_it[0], name="spikeLatencyMNIST_2.mp4")

    plt.close('all')
    plt.switch_backend('TkAgg')
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    ax = fig.add_subplot(111)
    y = spike_data_2[:, 0].view(steps, -1)
    print(y.size())
    splt.raster(spike_data_2[:, 0].view(steps, -1), ax, s=25, c="black")
    ax.set_xlim(-10, len(spike_data_2))
    ax.set_xticks(range(0, len(spike_data_2)+1, 10))  
    plt.title("Input Layer")
    plt.xlabel("Time step")
    plt.ylabel("Neuron Number")
    plt.show()

    return None


### Delta modulation  ###

'''
There are theories that the retina is adaptive: it will only process information when there is something new to process.
If there is no change in your field of view, then your photoreceptor cells are less prone to firing.

Biology is event-driven.

'''

def spikeDeltaModulationExample():
    print(f'Create a tensor with some fake time-series data')
    data = torch.Tensor([0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0])
    # Convert data
    spike_data = spikegen.delta(data, threshold=3, off_spike=False)

    plt.close('all')
    plt.switch_backend('TkAgg')

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # ax1 = fig.add_subplot(121)
    ax1.plot(data, color='blue')
    ax1.set_title("Time-series data")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Voltage (mV)")
    ax1.set_xticks(range(len(data)))

    # ax2 = fig.add_subplot(122)
    splt.raster(spike_data, ax2, c="black")
    ax2.set_title("Spike Raster")
    ax2.set_ylabel("Neuron Number")
    ax2.set_xlabel("Time step")
    plt.show()


if __name__ == "__main__":
    # spikeRateEncodingExample(100, 0.3)
    # spikeRateEncodingMNIST(train_loader, numberOfSteps, False)
    # spikeLatencyEncodingExample()
    # spikeLatencyEncodingMNIST()
    spikeDeltaModulationExample()
    print("Hvala!")