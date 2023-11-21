import sys
sys.path.append('../')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def randomForestClassification(x_train, x_test, y_train, y_test, mode):
     # Classification using Random forest
    # Make random forest model
    
    # Training
    start = time()
    randomForestClassifier = RandomForestClassifier()
    randomForestClassifier.fit(x_train, y_train)
    
    # Predict
    y_pred = randomForestClassifier.predict(x_test)
    stop = time()

    # Print report 
    if mode == 'binary':
        printResults.printResults(y_test, y_pred, "Random Forest", 'binary')
    elif mode == 'multi':
        printResults.printResults(y_test, y_pred, "Random Forest", 'macro')
    print(f'Time spent is {(stop - start):.3f} seconds.')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def randomForestOptimized(x_train, x_test, y_train, y_test, n, nClass):

    # Make random forest model
    start = time()
    randomForestClassifier = RandomForestClassifier(n_estimators = n)
    randomForestClassifier.fit(x_train, y_train)
    
    # Predict
    y_pred = randomForestClassifier.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Random Forest", nClass)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def randomForestCV(x, y, mode):

    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        randomForestClassifier = RandomForestClassifier()
        scores = cross_validate(randomForestClassifier, x, y, scoring = scores, cv = 5)
        print("Random forest:")
        print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi': #provjeriti za AP
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        randomForestClassifier = RandomForestClassifier()
        scores = cross_validate(randomForestClassifier, x, y, scoring = scores, cv = 5)
        print("Random forest:")
        print(scores)
        AP = scores['test_average_precision']
        return AP    


def rfNumberOfEstimators(x_train, x_test, y_train, n):

    randomForestClassifier = RandomForestClassifier(n_estimators = n)
    randomForestClassifier.fit(x_train, y_train)
    y_pred = randomForestClassifier.predict(x_test)
    return y_pred