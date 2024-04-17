'''
Support Vector Machine models for classification, i.e. Support Vector Classifiers
'''

import sys
sys.path.append('../')

from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def supportVectorMachineClassification(x_train, x_test, y_train, y_test, kernel = 'linear', mode = 'binary', verbose = 0):

    # Make Support Vector Machine model for classification
    start = time()
    svmClass = svm.SVC(kernel = kernel, max_iter = 50000)
    svmClass.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmClass.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    stop = time()

    # Print report 
    if verbose:
        if mode == 'binary':
            printResults.printResults(y_test, y_pred, kernel + "SVM", 'binary')
        elif mode == 'multi':
            printResults.printResults(y_test, y_pred, kernel + "SVM", 'macro')
        print(f'Time spent is {(stop - start):.3f} seconds.')
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return acc 


def svmC(x_train, x_test, y_train, kernel, c):

    svmC = svm.SVC(kernel = kernel, max_iter = 50000, C = c)
    svmC.fit(x_train, y_train)
    y_pred = svmC.predict(x_test)
    return y_pred


def svmOptimized(x_train, x_test, y_train, y_test, kernel, c, nClass):

    # Make Support Vector Machine model
    start = time()
    svmKernel = svm.SVC(kernel = kernel, max_iter = 50000, C = c)
    svmKernel.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmKernel.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, kernel + " SVM", nClass)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return None 


def supportVectorMachineCVOptimal1(x, y, kernel, c):

    # SVM classification with optimized parameters for Credit card fraud detection problem 
    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
    svmKernel = svm.SVC(kernel = kernel, max_iter = 50000, C = c)
    scores = cross_validate(svmKernel, x, y, scoring = scores, cv = 5)
    print(kernel + " SVM (best performance):")
    AP = scores['test_average_precision']
    return AP 


def supportVectorMachineCV(x, y, kernel = 'linear', mode = 'binary', verbose = 0):

    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        svmKernel = svm.SVC(kernel = kernel, max_iter = 10000)
        scores = cross_validate(svmKernel, x, y, scoring = scores, cv = 5)
        if verbose:
            print(kernel + " SVM:")
            print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi': 
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']
        svmKernel = svm.SVC(kernel = kernel, max_iter = 10000)
        scores = cross_validate(svmKernel, x, y, scoring = scores, cv = 5)
        if verbose:
            print(kernel + " SVM:")
            print(scores)
        P = scores['test_precision_macro']
        return P  

