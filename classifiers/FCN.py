import sys
sys.path.append('../')

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from classifiers import printResults
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate


def FullConnectedNetwork1(N):

    # Make fully connected network with N inputs and binary classification
    model = keras.Sequential()
    model.add(layers.Input(N, ))
    model.add(layers.Dense(4, activation = 'relu'))
    model.add(layers.Dense(4, activation = 'relu'))
    model.add(layers.Dense(1, activation = 'sigmoid'))
    # model.summary()

    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model


def TrainFCN(x_train, y_train, model, makeFigures):

    # Train the FCN 
    history = model.fit(x_train, y_train, batch_size = 128, epochs = 10, validation_split = 0.1, verbose = 0)
    model.save_weights('../networks/model1.h5')
    if (makeFigures == 1):
        lossTrain = history.history['loss']
        lossValid = history.history['val_loss']
        accTrain = history.history['accuracy']
        accValid = history.history['val_accuracy']

        plt.figure()
        plt.title("Train/valid loss")
        plt.plot(lossTrain, label = "Train loss")
        plt.plot(lossValid, label = "Validation loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        plt.show()
        
        plt.figure()
        plt.title("Train/valid accuracy")
        plt.plot(accTrain, label = "Train accuracy")
        plt.plot(accValid, label = "Validation accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    return model 


def PredictFCN(x_test, y_test, model):

    # Evaluate the network
    # lossTest, accTest = model.evaluate(x_test, y_test, verbose = 0)
    # print("Loss on test is ", lossTest)
    # print("Accuracy on test is ", accTest)
    predictionTest = model.predict(x_test)
    y_pred = [0 if i < 0.5 else 1 for i in predictionTest]
    printResults.printResults(y_test, y_pred, "Fully Connected Network", 'binary')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))


def FCNmodel1CV(x, y):
    
    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Load model
    model = FullConnectedNetwork1(29)
    model.load_weights('../networks/model1.h5')
    scores = cross_validate(model, x, y, scoring = scores, cv = 5)
    print("Fully connected network model1:")
    # print(f'Time spent is {stop - start} seconds.', sep='')
    AP = scores['test_average_precision']
    return AP 


