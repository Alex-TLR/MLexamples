# Credit Card Fraud Detection

The categories are highly unballanced. The are 284 315 valid and 492 fraud transactions.  
There is a random train/test split, therefore the results could be different for each run.  
Values V1, V2, V3 ... V28 and Amount are used as input, and Class is used as an output. Since Class can take values 0 for valid transaction or 1 for fraud transaction, only input data is standardized to have zero mean and unit standard deviation.  

The report for the run with random train/test split (with random_state = 42):


Valid cases: 284315  
Fraud cases: 492  
Number of test samples 56962  
Valid test cases: 56864  
Fraud test cases: 98     


Time spent is 0.2623012065887451 seconds.  
Number of errors for Linear Regression model is 64  
Linear Regression: Accuracy is 0.999  
Linear Regression: Precision is 0.827  
Linear Regression: Recall is 0.439  
Linear Regression: F1-Score is 0.573  
Linear Regression: Matthews correlation coefficient is 0.602  
Confusion matrix:  
 [[56855     9]  
 [   55    43]]  


Time spent is 0.9584383964538574 seconds.  
Number of errors for Logistic Regression model is 50  
Logistic Regression: Accuracy is 0.999  
Logistic Regression: Precision is 0.864  
Logistic Regression: Recall is 0.582  
Logistic Regression: F1-Score is 0.695  
Logistic Regression: Matthews correlation coefficient is 0.708  
Confusion matrix:  
 [[56855     9]  
 [   41    57]]  


Time spent is 2.8753137588500977 seconds.  
Number of errors for Decision Tree model is 41  
Decision Tree: Accuracy is 0.999  
Decision Tree: Precision is 0.870  
Decision Tree: Recall is 0.684  
Decision Tree: F1-Score is 0.766  
Decision Tree: Matthews correlation coefficient is 0.771  
Confusion matrix:  
 [[56854    10]  
 [   31    67]]  


Time spent is 13.502797842025757 seconds.  
Number of errors for kNN model is 25  
kNN: Accuracy is 1.000  
kNN: Precision is 0.940  
kNN: Recall is 0.796    
kNN: F1-Score is 0.862  
kNN: Matthews correlation coefficient is 0.865  
Confusion matrix:  
 [[56859     5]  
 [   20    78]]  


Time spent is 167.79352736473083 seconds.  
Number of errors for Random Forest model is 22  
Random Forest: Accuracy is 1.000  
Random Forest: Precision is 0.963  
Random Forest: Recall is 0.806  
Random Forest: F1-Score is 0.878  
Random Forest: Matthews correlation coefficient is 0.881  
Confusion matrix:  
 [[56861     3]  
 [   19    79]]  


Time spent is 7404.125505685806 seconds.  
Number of errors for linear Support Vector Machine model is 38  
linear Support Vector Machine: Accuracy is 0.999  
linear Support Vector Machine: Precision is 0.819  
linear Support Vector Machine: Recall is 0.786  
linear Support Vector Machine: F1-Score is 0.802  
linear Support Vector Machine: Matthews correlation coefficient is 0.802  
Confusion matrix:  
 [[56847    17]  
 [   21    77]]  


Time spent is 225.81356525421143 seconds.  
Number of errors for rbf Support Vector Machine model is 38  
rbf Support Vector Machine: Accuracy is 0.999  
rbf Support Vector Machine: Precision is 0.819  
rbf Support Vector Machine: Recall is 0.786  
rbf Support Vector Machine: F1-Score is 0.802  
rbf Support Vector Machine: Matthews correlation coefficient is 0.802  
Confusion matrix:  
 [[56847    17]  
 [   21    77]]  
