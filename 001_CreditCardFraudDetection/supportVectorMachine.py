from sklearn import svm
from sklearn.metrics import confusion_matrix
import printResults
from time import time

def supportVectorMachine(x_train, x_test, y_train, y_test):

    # Make linear Support Vector Machine model
    start = time()
    svmLinear = svm.SVC(kernel='linear')
    svmLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred1 = svmLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred1, "linear Support Vector Machine")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred1))
    print("\n")

    # Make non-linear Support Vector Machine model
    start = time()
    svmNonLinear = svm.SVC(kernel='rbf')
    svmNonLinear.fit(x_train, y_train)
    
    # Predict 
    y_pred2 = svmLinear.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred2, "rbf Support Vector Machine")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred2))
    print("\n")

    return None 
