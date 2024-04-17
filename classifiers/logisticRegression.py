import sys
sys.path.append('../')

from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def logisticRegressionClassification(x_train, x_test, y_train, y_test, mode = 'binary', verbose = 0):
    # Classification using logistic regression
    # Make logistic regression model
    # Time includes both training and prediction

    # Training
    start = time()
    LogRegression = linear_model.LogisticRegression(max_iter = 1000)
    LogRegModel = LogRegression.fit(x_train, y_train)
    # Predict
    y_pred = LogRegModel.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    stop = time()

    # Print report 
    if verbose:
        if mode == 'binary':
            printResults.printResults(y_test, y_pred, "Logistic Regression", 'binary')
        elif mode == 'multi':
            printResults.printResults(y_test, y_pred, "Logistic Regression", 'macro')
        print(f'Time spent is {(stop - start):.3f} seconds.')
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return acc 


def logisticRegressionCV(x, y, mode = 'binary', verbose = 0):

    # Make logistic regression model
    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        LogRegression = linear_model.LogisticRegression(multi_class = 'ovr', max_iter = 500)
        scores = cross_validate(LogRegression, x, y, scoring = scores, cv = 5)
        if verbose:
            print("Logistic regression:")
            print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']
        LogRegression = linear_model.LogisticRegression(multi_class = 'multinomial', max_iter = 500)
        scores = cross_validate(LogRegression, x, y, scoring = scores, cv = 5)
        if verbose:
            print("Logistic regression:")
            print(scores)
        P = scores['test_precision_macro']
        return P 
