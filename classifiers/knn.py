import sys
sys.path.append('../')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time


def kNearestNeighborsClassification(x_train, x_test, y_train, y_test, mode = 'binary', verbose = 0):
    # Classification using KNN
    # Make KNN model

    # Training
    start = time()
    kNN = KNeighborsClassifier()
    kNN.fit(x_train, y_train)
    # Predict
    y_pred = kNN.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    stop = time()

    # Print report
    if verbose:
        if mode == 'binary':
            printResults.printResults(y_test, y_pred, "kNN", 'binary')
        elif mode == 'multi':
            printResults.printResults(y_test, y_pred, "kNN", 'macro')
        print(f'Time spent is {(stop - start):.3f} seconds.')
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    return acc


def knnOptimized(x_train, x_test, y_train, y_test, n, nClass):

    # Make KNN model
    start = time()
    kNN = KNeighborsClassifier(n_neighbors = n)
    kNN.fit(x_train, y_train)
    
    # Predict
    y_pred = kNN.predict(x_test)
    stop = time()

    # Print report
    printResults.printResults(y_test, y_pred, "kNN", nClass)
    print(f'Time spent is {stop - start} seconds.')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None


def kNearestNeighborsCV(x, y, mode = 'binary', verbose = 0):

    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        kNN = KNeighborsClassifier()
        scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
        if verbose:
            print("k Nearest neighbors:")
            print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi': 
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef']
        kNN = KNeighborsClassifier()
        scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
        if verbose:
            print("k Nearest neighbors:")
            print(scores)
        P = scores['test_precision_macro']
        return P 


def kNearestNeighborsCVOptimal1(x, y):

    # kNN classification with optimized parameters for Credit card fraud detection problem   
    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
    kNN = KNeighborsClassifier(n_neighbors = 7)
    scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
    print("k Nearest neighbors (best performance):")
    AP = scores['test_average_precision']
    return AP 


# def knnAlgorithm(x_train, x_test, y_train, a):

#     kNN = KNeighborsClassifier(algorithm = a)
#     kNN.fit(x_train, y_train)
#     y_pred = kNN.predict(x_test)
#     return y_pred


# def knnDistance(x_train, x_test, y_train, a, p):

#     kNN = KNeighborsClassifier(algorithm = a, p = p)
#     kNN.fit(x_train, y_train)
#     y_pred = kNN.predict(x_test)
#     return y_pred


def knnNumberOfNeighbors(x_train, x_test, y_train, n):

    kNN = KNeighborsClassifier(n_neighbors = n)
    kNN.fit(x_train, y_train)
    y_pred = kNN.predict(x_test)
    return y_pred
