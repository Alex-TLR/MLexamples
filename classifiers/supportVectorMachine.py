'''
Support Vector Machine models for classification, i.e. Support Vector Classifiers
'''

import sys
sys.path.append('../')

from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def supportVectorMachineLinearBinary(x_train, x_test, y_train, y_test):

    # Make linear Support Vector Machine model for classification
    start = time()
    svmLinear = svm.SVC(kernel='linear', max_iter = 10000)
    svmLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Linear SVM", 'binary')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def supportVectorMachineLinearMulticlass(x_train, x_test, y_train, y_test):

    # Make linear Support Vector Machine model
    start = time()
    svmLinear = svm.SVC(kernel='linear', max_iter = 10000)
    svmLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Linear SVM", 'macro')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def svmC(x_train, x_test, y_train, kernel, c):

    svmC = svm.SVC(kernel = kernel, max_iter = 10000, C = c)
    svmC.fit(x_train, y_train)
    y_pred = svmC.predict(x_test)
    return y_pred


def svmOptimized(x_train, x_test, y_train, y_test, kernel, c, nClass):

    # Make Support Vector Machine model
    start = time()
    svmKernel = svm.SVC(kernel = kernel, max_iter = 10000, C = c)
    svmKernel.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmKernel.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, kernel + " SVM", nClass)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def supportVectorMachineCVOptimal(x, y, kernel, c):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make logistic regression model
    start = time()
    svmKernel = svm.SVC(kernel = kernel, max_iter = 10000, C = c)
    scores = cross_validate(svmKernel, x, y, scoring = scores, cv = 5)
    stop = time()
    print(kernel + " SVM (best performance):")
    print(f'Time spent is {stop - start} seconds.', sep="")
    AP = scores['test_average_precision']
    return AP 


def supportVectorMachineRbfBinary(x_train, x_test, y_train, y_test):

    # Make non-linear Support Vector Machine model
    start = time()
    svmNonLinear = svm.SVC(kernel='rbf', max_iter = 10000)
    svmNonLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmNonLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "rbf SVM", 'binary')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def supportVectorMachineRbfMulticlass(x_train, x_test, y_train, y_test):

    # Make non-linear Support Vector Machine model
    start = time()
    svmNonLinear = svm.SVC(kernel='rbf', max_iter = 10000)
    svmNonLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred = svmNonLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "rbf SVM", 'macro')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def supportVectorMachineCV(x, y, kernel):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make linear SVM model
    start = time()
    svmKernel = svm.SVC(kernel = kernel, max_iter = 10000)
    scores = cross_validate(svmKernel, x, y, scoring = scores, cv = 5)
    stop = time()
    print(kernel + " SVM:")
    print(f'Time spent is {stop - start} seconds.', sep='')
    AP = scores['test_average_precision']
    return AP 


def supportVectorMachineRbfCV(x, y):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make non-linear SVM model
    start = time()
    svmNonLinear = svm.SVC(kernel='rbf', max_iter = 10000)
    scores = cross_validate(svmNonLinear, x, y, scoring = scores, cv = 5)
    stop = time()
    print("rbf SVM:")
    print(f'Time spent is {stop - start} seconds.', sep='')
    AP = scores['test_average_precision']
    return AP 

