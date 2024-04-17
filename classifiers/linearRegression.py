import sys
sys.path.append('../')

from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from classifiers import printResults
from time import time

def linearRegressionClassification(x_train, x_test, y_train, y_test, mode = 'binary', verbose = 0):
    
    # Classification using Linear regression
    # Make linear regression model
    # Time includes both training and prediction

    # Training
    start = time()
    LinearRegression = linear_model.LinearRegression()
    LinRegModel = LinearRegression.fit(x_train, y_train)
    # Predict 
    y_pred = LinRegModel.predict(x_test)
    if mode == 'binary':
        y_pred = [0 if i <= 0.5 else 1 for i in y_pred]
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            printResults.printResults(y_test, y_pred, "Linear Regression", 'binary')
    elif mode == 'multi':
        for i in range(len(y_pred)):
            y_pred[i] = [0 if i <= 0.5 else 1 for i in y_pred[i]]
        y_predN = np.argmax(y_pred, axis = 1)
        y_testN = np.argmax(y_test, axis = 1)
        y_pred = y_predN 
        y_test = y_testN
        acc = accuracy_score(y_test, y_pred)
        if verbose:
            printResults.printResults(y_test, y_pred, "Linear Regression", 'macro')
    stop = time()

    # Print report
    if verbose:
        print(f'Time spent is {(stop - start):.3f} seconds.')
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return acc     