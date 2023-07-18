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


Time spent is 0.8287839889526367 seconds.  
Number of errors for Logistic Regression model is 39  
Logistic Regression: Accuracy is 0.999  
Logistic Regression: Precision is 0.875  
Logistic Regression: Recall is 0.560  
Logistic Regression: F1-Score is 0.683  
Logistic Regression: Matthews correlation coefficient is 0.700  
Confusion matrix:  
 [[56881     6]  
 [   33    42]]   


Time spent is 13.57843804359436 seconds.  
Number of errors for Decision Tree model is 56  
Decision Tree: Accuracy is 0.999  
Decision Tree: Precision is 0.617  
Decision Tree: Recall is 0.667  
Decision Tree: F1-Score is 0.641  
Decision Tree: Matthews correlation coefficient is 0.641  
Confusion matrix:  
 [[56856    31]  
 [   25    50]]   


Time spent is 9.292028427124023 seconds.  
Number of errors for kNN model is 23  
kNN: Accuracy is 1.000  
kNN: Precision is 0.948  
kNN: Recall is 0.733  
kNN: F1-Score is 0.827  
kNN: Matthews correlation coefficient is 0.834  
Confusion matrix:  
 [[56884     3]  
 [   20    55]]   


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


Time spent is 32.01217436790466 seconds.  
Number of errors for Linear SVM model is 42  
Linear SVM: Accuracy is 0.999  
Linear SVM: Precision is 1.000  
Linear SVM: Recall is 0.440  
Linear SVM: F1-Score is 0.611  
Linear SVM: Matthews correlation coefficient is 0.663  
Confusion matrix:  
 [[56887     0]  
 [   42    33]]  


Time spent is 293.95291686058044 seconds.  
Number of errors for rbf SVM model is 31  
rbf SVM: Accuracy is 0.999  
rbf SVM: Precision is 0.978  
rbf SVM: Recall is 0.600  
rbf SVM: F1-Score is 0.744  
rbf SVM: Matthews correlation coefficient is 0.766  
Confusion matrix:  
 [[56886     1]  
 [   30    45]]  
