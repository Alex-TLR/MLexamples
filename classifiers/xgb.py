import sys
sys.path.append('../')

from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def xgboostClassification(x_train, x_test, y_train, y_test, mode = 'binary', verbose = 0):
    # Classification using XGBClassifier
    # Time includes both training and prediction

    # Training
    start = time()
    xgb = XGBClassifier()
    xgbModel = xgb.fit(x_train, y_train)
    # Predict
    y_pred = xgbModel.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    stop = time()

    # Print report 
    if verbose:
        if mode == 'binary':
            printResults.printResults(y_test, y_pred, "XGB Classifier", 'binary')
        elif mode == 'multi':
            printResults.printResults(y_test, y_pred, "XGB Classifier", 'macro')
        print(f'Time spent is {(stop - start):.3f} seconds.')
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return acc 


def xgboostCV(x, y, mode = 'binary', verbose = 0):

    # Make XGBClassifier model
    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        xgbModel = XGBClassifier()
        scores = cross_validate(xgbModel, x, y, scoring = scores, cv = 5)
        if verbose:
            print("XGB Classifier:")
            print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']
        xgbModel = XGBClassifier()
        scores = cross_validate(xgbModel, x, y, scoring = scores, cv = 5)
        if verbose:
            print("XGB Classifier:")
            print(scores)
        P = scores['test_precision_macro']
        return P 
