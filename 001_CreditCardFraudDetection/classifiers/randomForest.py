from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import printResults
from time import time

def randomForest(x_train, x_test, y_train, y_test):

    # Make random forest model
    start = time()
    randomForestClassifier = RandomForestClassifier()
    randomForestClassifier.fit(x_train, y_train)
    
    # Predict
    y_pred = randomForestClassifier.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Random Forest")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def randomForestCV(x, y):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
    
    # Make random tree model
    start = time()
    randomForestClassifier = RandomForestClassifier()
    scores = cross_validate(randomForestClassifier, x, y, scoring = scores, cv = 5)
    stop = time()
    print("Random forest:")
    print(f'Time spent is {stop - start} seconds.', sep='')
    AP = scores['test_average_precision']
    return AP     