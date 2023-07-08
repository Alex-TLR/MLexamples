from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import printResults
from time import time

def logisticRegression(x_train, x_test, y_train, y_test):

    # Make logistic regression model
    start = time()
    LogRegression = linear_model.LogisticRegression(multi_class='ovr', max_iter = 1000)
    LogRegModel = LogRegression.fit(x_train, y_train)
    
    # Predict
    y_pred = LogRegModel.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Logistic Regression")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 
