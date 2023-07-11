from sklearn import tree
from sklearn.metrics import confusion_matrix
import printResults
from time import time 

def decisionTree(x_train, x_test, y_train, y_test):

    # Make decision tree model
    start = time()
    decisionTree = tree.DecisionTreeClassifier()
    decisionTree.fit(x_train, y_train)
    
    # Predict 
    y_pred = decisionTree.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Decision Tree")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 


def dt(x_train, x_test, y_train, y_test, c, maxDepth, maxLeafNodes):
    
    # Make decision tree model
    start = time()
    decisionTree = tree.DecisionTreeClassifier(criterion = c, max_depth = maxDepth, max_leaf_nodes = maxLeafNodes)
    decisionTree.fit(x_train, y_train)
    
    # Predict 
    y_pred = decisionTree.predict(x_test)
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report 
    printResults.printResults(y_test, y_pred, "Decision Tree")
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
