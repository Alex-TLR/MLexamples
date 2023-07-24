import sys
sys.path.append('../')

from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import numpy as np
from classifiers import printResults
from time import time

def linearRegression(x_train, x_test, y_train, y_test, ohe):

    # Make linear regression model
    start = time()
    LinearRegression = linear_model.LinearRegression()
    LinRegModel = LinearRegression.fit(x_train, y_train)
    
    # Predict 
    y_pred = LinRegModel.predict(x_test)
    if (ohe == 0):
        y_pred = [0 if i <= 0.5 else 1 for i in y_pred]
    elif (ohe == 1):
        for i in range(len(y_pred)):
            y_pred[i] = [0 if i <= 0.5 else 1 for i in y_pred[i]]
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Reverse one hot encoding (if necessary)
    if (ohe == 1):
        y_predN = np.argmax(y_pred, axis = 1)
        y_testN = np.argmax(y_test, axis = 1)
        y_pred = y_predN 
        y_test = y_testN

    # Print report
    printResults.printResults(y_test, y_pred, "Linear Regression")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 

