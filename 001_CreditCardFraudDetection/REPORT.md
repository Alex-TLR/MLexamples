# Credit Card Fraud Detection

The categories are highly unballanced. The are 284 315 valid and 492 fraud transactions.  
There is a random train/test split, therefore the results could be different for each run.  
Values V1, V2, V3 ... V28 and Amount are used as input, and Class is used as an output. Since Class can take values 0 for valid transaction or 1 for fraud transaction, only input data is standardized to have zero mean and unit standard deviation.  

The report for the run with non-shuffled train/test split :


Valid cases: 284315  
Fraud cases: 492  
Number of test samples 56962  
Valid test cases: 56887  
Fraud test cases: 75  


Time spent is 0.7003524303436279 seconds.  
Number of errors for Linear Regression model is 66  
Linear Regression: Accuracy is 0.999  
Linear Regression: Precision is 1.000  
Linear Regression: Recall is 0.120  
Linear Regression: F1-Score is 0.214  
Linear Regression: Matthews correlation coefficient is 0.346  
Confusion matrix:  
 [[56887     0]  
 [   66     9]]  


Time spent is 1.5493683815002441 seconds.  
Number of errors for Logistic Regression model is 59  
Logistic Regression: Accuracy is 0.999  
Logistic Regression: Precision is 0.944  
Logistic Regression: Recall is 0.227  
Logistic Regression: F1-Score is 0.366  
Logistic Regression: Matthews correlation coefficient is 0.462  
Confusion matrix:  
 [[56886     1]  
 [   58    17]]  


Time spent is 15.130817651748657 seconds.  
Number of errors for Decision Tree model is 51  
Decision Tree: Accuracy is 0.999  
Decision Tree: Precision is 0.654  
Decision Tree: Recall is 0.680  
Decision Tree: F1-Score is 0.667  
Decision Tree: Matthews correlation coefficient is 0.666  
Confusion matrix:  
 [[56860    27]  
 [   24    51]]  


Time spent is 13.854992389678955 seconds.  
Number of errors for kNN model is 37  
kNN: Accuracy is 0.999  
kNN: Precision is 0.952  
kNN: Recall is 0.533  
kNN: F1-Score is 0.684  
kNN: Matthews correlation coefficient is 0.712  
Confusion matrix:  
 [[56885     2]  
 [   35    40]]  


Time spent is 163.76197052001953 seconds.  
Number of errors for Random Forest model is 24  
Random Forest: Accuracy is 1.000  
Random Forest: Precision is 0.981  
Random Forest: Recall is 0.693  
Random Forest: F1-Score is 0.812  
Random Forest: Matthews correlation coefficient is 0.825  
Confusion matrix:  
 [[56886     1]  
 [   23    52]]  


Time spent is 28.73442316055298 seconds.  
Number of errors for linear Support Vector Machine model is 24  
linear Support Vector Machine: Accuracy is 1.000  
linear Support Vector Machine: Precision is 1.000  
linear Support Vector Machine: Recall is 0.680  
linear Support Vector Machine: F1-Score is 0.810  
linear Support Vector Machine: Matthews correlation coefficient is 0.824  
Confusion matrix:  
 [[56887     0]  
 [   24    51]]  


Time spent is 295.2604675292969 seconds.  
Number of errors for rbf Support Vector Machine model is 24  
rbf Support Vector Machine: Accuracy is 1.000  
rbf Support Vector Machine: Precision is 1.000  
rbf Support Vector Machine: Recall is 0.680  
rbf Support Vector Machine: F1-Score is 0.810  
rbf Support Vector Machine: Matthews correlation coefficient is 0.824  
Confusion matrix:  
 [[56887     0]  
 [   24    51]]  
