import sys
sys.path.append('../')

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def GaussianNaiveBayes(x_train, x_test, y_train, y_test):

    # Classification using GaussianNaiveBayes
    # Make GaussianNB model
    start = time()
    GNB = GaussianNB()
    GNBModel = GNB.fit(x_train, y_train)
    
    # Predict 
    y_pred = GNBModel.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report
    printResults.printResults(y_test, y_pred, "Gaussian Navie Bayes")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def GaussianNaiveBayesCV(x, y):

    scores = ['average_precision']

    # Make logistic regression model
    start = time()
    GNB = GaussianNB()
    scores = cross_validate(GNB, x, y, scoring = scores, cv = 5)
    stop = time()
    print("Gaussian Naive Bayes:")
    print(f'Time spent is {stop - start:.3f} seconds.', sep="")
    AP = scores['test_average_precision']
    return AP 
