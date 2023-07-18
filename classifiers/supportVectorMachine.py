from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
import printResults
from time import time

def supportVectorMachineLinear(x_train, x_test, y_train, y_test):

    # Make linear Support Vector Machine model
    start = time()
    svmLinear = svm.SVC(kernel='linear', max_iter = 10000)
    svmLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred1 = svmLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred1, "Linear SVM")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred1))
    print("\n")
    return None 


def supportVectorMachineRbf(x_train, x_test, y_train, y_test):

    # Make non-linear Support Vector Machine model
    start = time()
    svmNonLinear = svm.SVC(kernel='rbf', max_iter = 10000)
    svmNonLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred2 = svmNonLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred2, "rbf SVM")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred2))
    print("\n")
    return None 


def supportVectorMachineLinearCV(x, y):

    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']

    # Make linear SVM model
    start = time()
    svmLinear = svm.SVC(kernel='linear', max_iter = 10000)
    scores = cross_validate(svmLinear, x, y, scoring = scores, cv = 5)
    stop = time()
    print("Linear SVM:")
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
