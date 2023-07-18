from sklearn import linear_model
from sklearn.metrics import confusion_matrix, get_scorer_names
from sklearn.model_selection import cross_validate
import printResults
from time import time

def logisticRegression(x_train, x_test, y_train, y_test):

    # Make logistic regression model
    start = time()
    LogRegression = linear_model.LogisticRegression()
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


def logisticRegressionCV(x, y):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make logistic regression model
    start = time()
    LogRegression = linear_model.LogisticRegression()
    scores = cross_validate(LogRegression, x, y, scoring = scores, cv = 5)
    stop = time()
    print("Logistic regression:")
    print(f'Time spent is {stop - start} seconds.', sep='')
    AP = scores['test_average_precision']
    return AP 
