import sys
sys.path.append('../')

from sklearn import linear_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def linearRegression(x_train, x_test, y_train, y_test):

    # Make linear regression model
    start = time()
    LinearRegression = linear_model.LinearRegression()
    LinRegModel = LinearRegression.fit(x_train, y_train)
    
    # Predict 
    y_pred = LinRegModel.predict(x_test)
    y_pred = [0 if i <= 0.5 else 1 for i in y_pred]
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report
    printResults.printResults(y_test, y_pred, "Linear Regression")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 

    
# def linearRegressionCV(x, y):

#     scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
    
#     # Make linear regression model
#     start = time()
#     LinearRegression = linear_model.LinearRegression()
#     scores = cross_validate(LinearRegression, x, y, scoring = scores)
#     stop = time()
# #     print(f'Time spent is {stop - start} seconds.')

# #     # Print report
# #     printResults.printResults(y_test, y_pred, "Linear Regression")
# #     print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
# #     print("\n")
# #     return None 
