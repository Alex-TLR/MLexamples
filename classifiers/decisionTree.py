import sys
sys.path.append('../')

from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from classifiers import printResults
from time import time 

def decisionTreeClassification(x_train, x_test, y_train, y_test, mode):
    # Classification using Decision Trees
    # Make decision tree model

    # Training
    start = time()
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree.fit(x_train, y_train) 
    # Predict 
    y_pred = decisionTree.predict(x_test)
    stop = time()
    
    # Print report 
    if mode == 'binary':
        printResults.printResults(y_test, y_pred, "Decision Tree", 'binary')
    elif mode == 'multi':
        printResults.printResults(y_test, y_pred, "Decision Tree", 'macro')
    print(f'Time spent is {(stop - start):.3f} seconds.')
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def dtOptimized(x_train, x_test, y_train, y_test, c, maxDepth, maxLeafNodes, nClass):
    
    # Make decision tree model
    start = time()
    decisionTree = tree.DecisionTreeClassifier(criterion = c, max_depth = maxDepth, max_leaf_nodes = maxLeafNodes)
    decisionTree.fit(x_train, y_train)
    
    # Predict 
    y_pred = decisionTree.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Decision Tree", nClass)
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def dtCriterion(x_train, x_test, y_train, c):
    
    decisionTree = tree.DecisionTreeClassifier(criterion = c)
    decisionTree.fit(x_train, y_train)
    y_pred = decisionTree.predict(x_test)
    return y_pred


def dtMaxDepth(x_train, x_test, y_train, c, maxDepth):

    decisionTree = tree.DecisionTreeClassifier(criterion = c, max_depth = maxDepth)
    decisionTree.fit(x_train, y_train)
    y_pred = decisionTree.predict(x_test)
    return y_pred


def dtMaxLeafNodes(x_train, x_test, y_train, c, maxDepth, maxLeafNodes):

    decisionTree = tree.DecisionTreeClassifier(criterion = c, max_depth = maxDepth, max_leaf_nodes = maxLeafNodes)
    decisionTree.fit(x_train, y_train)
    y_pred = decisionTree.predict(x_test)
    return y_pred


def decisionTreeCV(x, y, mode):

    # Make decision tree model
    if mode == 'binary':
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        decisionTree = tree.DecisionTreeClassifier()
        scores = cross_validate(decisionTree, x, y, scoring = scores, cv = 5)
        print("Decision trees:")
        print(scores)
        AP = scores['test_average_precision']
        return AP 

    elif mode == 'multi': #provjeriti zaAP
        scores = ['precision_macro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
        decisionTree = tree.DecisionTreeClassifier()
        scores = cross_validate(decisionTree, x, y, scoring = scores, cv = 5)
        print("Decision trees:")
        print(scores)
        AP = scores['test_average_precision']
        return AP 


def decisionTreeCVOptimal1(x, y):

    # Decision tree classification with optimized parameters for Credit card fraud detection problem
    scores = ['precision_micro', 'recall_macro', 'f1_macro', 'matthews_corrcoef', 'average_precision']
    decisionTree = tree.DecisionTreeClassifier(criterion = 'gini', max_depth = 4, max_leaf_nodes = 10)
    scores = cross_validate(decisionTree, x, y, scoring = scores, cv = 5)
    print("Decision trees (best performance):")
    AP = scores['test_average_precision']
    return AP 

