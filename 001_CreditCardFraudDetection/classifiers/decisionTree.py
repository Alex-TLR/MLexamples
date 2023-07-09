from sklearn import tree
from sklearn.metrics import confusion_matrix
import printResults
from time import time 

def decisionTree(x_train, x_test, y_train, y_test):

    # Make decision tree model
    start = time()
    decisionTree = tree.DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, max_leaf_nodes=5)
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
