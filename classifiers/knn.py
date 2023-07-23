import sys
sys.path.append('../')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time


def kNearestNeighbors(x_train, x_test, y_train, y_test):

    # Make KNN model
    start = time()
    kNN = KNeighborsClassifier()
    kNN.fit(x_train, y_train)
    
    # Predict
    y_pred = kNN.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report
    printResults.printResults(y_test, y_pred, "kNN")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None


def knnOptimized(x_train, x_test, y_train, y_test, n):

    # Make KNN model
    start = time()
    kNN = KNeighborsClassifier(n_neighbors = n)
    kNN.fit(x_train, y_train)
    
    # Predict
    y_pred = kNN.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report
    printResults.printResults(y_test, y_pred, "kNN")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None


def kNearestNeighborsCV(x, y):

    scores = ['average_precision']

    # Make logistic regression model
    start = time()
    kNN = KNeighborsClassifier()
    scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
    stop = time()
    print("k Nearest neighbors:")
    print(f'Time spent is {stop - start:.3f} seconds.', sep="")
    AP = scores['test_average_precision']
    return AP 


def kNearestNeighborsCVOptimal(x, y):

    scores = ['average_precision']

    # Make logistic regression model
    start = time()
    kNN = KNeighborsClassifier(n_neighbors = 7)
    scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
    stop = time()
    print("k Nearest neighbors:")
    print(f'Time spent is {stop - start:.3f} seconds.', sep="")
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
