from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
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
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 
