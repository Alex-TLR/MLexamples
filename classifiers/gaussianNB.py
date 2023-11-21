import sys
sys.path.append('../')

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time

def GaussianNaiveBayesClassification(x_train, x_test, y_train, y_test, mode):
    # Classification using Gaussian Naive Bayes
    # Time includes both training and prediction

    # Training
    start = time()
    GNB = GaussianNB()
    GNBModel = GNB.fit(x_train, y_train)
    # Predict
    y_pred = GNBModel.predict(x_test)
    stop = time()
    
    # mode should be set to binary, micro or macro by user
    # Print report 
    if mode == 'binary':
        printResults.printResults(y_test, y_pred, "Gaussian Navie Bayes", 'binary')
    elif mode == 'multi':
        printResults.printResults(y_test, y_pred, "Gaussian Navie Bayes", 'macro')
    print(f'Time spent is {(stop - start):.3f} seconds.')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def GaussianNaiveBayesCV(x, y, mode):

    if mode == 'binary': # provjeriti
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        GNB = GaussianNB()
        scores = cross_validate(GNB, x, y, scoring = scores, cv = 5)
        print("Gaussian Naive Bayes:")
        print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi': #provjeriti za AP
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        GNB = GaussianNB()
        scores = cross_validate(GNB, x, y, scoring = scores, cv = 5)
        print("Gaussian Naive Bayes:")
        print(scores)
        AP = scores['test_average_precision']
        return AP 
