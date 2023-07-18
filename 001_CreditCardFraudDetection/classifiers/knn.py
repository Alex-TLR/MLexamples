from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import printResults
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


def kNearestNeighborsCV(x, y):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make logistic regression model
    start = time()
    kNN = KNeighborsClassifier()
    scores = cross_validate(kNN, x, y, scoring = scores, cv = 5)
    stop = time()
    print("k Nearest neighbors:")
    print(f'Time spent is {stop - start} seconds.', sep="")
    AP = scores['test_average_precision']
    return AP 
