# Credit Card Fraud Detection

The categories are highly unballanced. The are 284 315 valid and 492 fraud transactions.  
There is a random train/test split, therefore the results could be different for each run.  
Values V1, V2, V3 ... V28 and Amount are used as input, and Class is used as an output. Since Class can take values 0 for valid transaction or 1 for fraud transaction, only input data is standardized to have zero mean and unit standard deviation.  

The report for the run with random train/test split:

Valid cases: 284315  
Fraud cases: 492  
Number of test samples 56962  
Valid test cases: 56865  
Fraud test cases: 97  


Time spent is 0.28099918365478516 seconds.  
Number of errors for Linear Regression model is 69   
Linear Regression: Accuracy is 0.999   
Linear Regression: Precision is 0.750   
Linear Regression: Recall is 0.433   
Linear Regression: F1-Score is 0.549   
Linear Regression: Matthews correlation coefficient is 0.569   
Confusion matrix:   
 [[56851    14]   
 [   55    42]]   


Time spent is 1.3414480686187744 seconds.  
Number of errors for Logistic Regression model is 67  
Logistic Regression: Accuracy is 0.999  
Logistic Regression: Precision is 0.759  
Logistic Regression: Recall is 0.454  
Logistic Regression: F1-Score is 0.568  
Logistic Regression: Matthews correlation coefficient is 0.586  
Confusion matrix:  
 [[56851    14]  
 [   53    44]]  


Time spent is 2.781132459640503 seconds.  
Number of errors for Decision Tree model is 48  
Decision Tree: Accuracy is 0.999  
Decision Tree: Precision is 0.763  
Decision Tree: Recall is 0.732  
Decision Tree: F1-Score is 0.747  
Decision Tree: Matthews correlation coefficient is 0.747  
Confusion matrix:  
 [[56843    22]  
 [   26    71]]  


Time spent is 13.665133714675903 seconds.  
Number of errors for kNN model is 34  
kNN: Accuracy is 0.999  
kNN: Precision is 0.932  
kNN: Recall is 0.701  
kNN: F1-Score is 0.800  
kNN: Matthews correlation coefficient is 0.808  
Confusion matrix:  
 [[56860     5]  
 [   29    68]]  


Time spent is 175.03132319450378 seconds.  
Number of errors for Random Forest model is 22  
Random Forest: Accuracy is 1.000  
Random Forest: Precision is 0.941  
Random Forest: Recall is 0.825  
Random Forest: F1-Score is 0.879  
Random Forest: Matthews correlation coefficient is 0.881  
Confusion matrix:  
 [[56860     5]  
 [   17    80]]  


Time spent is 121.82730627059937 seconds.  
Number of errors for linear Support Vector Machine model is 37  
linear Support Vector Machine: Accuracy is 0.999  
linear Support Vector Machine: Precision is 0.800  
linear Support Vector Machine: Recall is 0.825  
linear Support Vector Machine: F1-Score is 0.812  
linear Support Vector Machine: Matthews correlation coefficient is 0.812  
Confusion matrix:  
 [[56845    20]  
 [   17    80]]  


Time spent is 375.1610486507416 seconds.  
Number of errors for rbf Support Vector Machine model is 37  
rbf Support Vector Machine: Accuracy is 0.999  
rbf Support Vector Machine: Precision is 0.800  
rbf Support Vector Machine: Recall is 0.825  
rbf Support Vector Machine: F1-Score is 0.812  
rbf Support Vector Machine: Matthews correlation coefficient is 0.812  
Confusion matrix:  
 [[56845    20]  
 [   17    80]]  