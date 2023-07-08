from sklearn.metrics import accuracy_score 
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef

def printResults(y_test, y_pred, methodName):
    nErrors = (y_pred != y_test).sum()
    print(f'Number of errors for {methodName} model is {nErrors}')
    acc = accuracy_score(y_test, y_pred)
    print(f'{methodName}: Accuracy is {acc}')
    prec = precision_score(y_test, y_pred)
    print(f'{methodName}: Precision is {prec}')
    rec = recall_score(y_test, y_pred)
    print(f'{methodName}: Recall is {rec}')
    f1 = f1_score(y_test, y_pred)
    print(f'{methodName}: F1-Score is {f1}')
    MCC = matthews_corrcoef(y_test, y_pred)
    print(f'{methodName}: Matthews correlation coefficient is {MCC}')

    return None 