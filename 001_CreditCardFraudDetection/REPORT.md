# Credit Card Fraud Detection

The categories are highly unbalanced. The are 284 315 valid and 492 fraud transactions.  
There is a random train/test split, therefore the results could be different for each run.  
Values V1, V2, V3 ... V28, and Amount are input, and Class is output. Since Class can take values 0 for valid or 1 for fraud transactions, only input data is standardized to have zero mean and unit standard deviation.  

The report for the run with non-shuffled train/test split and standardized input data:


Valid cases: 284315  
Fraud cases: 492  
Number of test samples 56962  
Valid test cases: 56887  
Fraud test cases: 75  

| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Time (sec) | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------:|:----------------:|
|Linear regression   | 0.999    | 1.000     | 0.120  | 0.214    | 0.346 | 0.7003524  | 66               |
|Logistic regression | 0.999    | 0.875     | 0.560  | 0.683    | 0.700 | 0.8287839  | 39               |
|Decision trees      | 0.999    | 0.617     | 0.667  | 0.641    | 0.641 | 13.578438  | 56               |
|k Nearest Neighbors | 1.000    | 0.948     | 0.733  | 0.827    | 0.834 | 9.2920284  | 23               |
|Random forest       | 1.000    | 0.981     | 0.693  | 0.812    | 0.825 | 163.76197  | 24               |
|linear SVM          | 0.999    | 1.000     | 0.440  | 0.611    | 0.663 | 32.012174  | 42               |
|rbf SVM             | 0.999    | 0.978     | 0.600  | 0.744    | 0.766 | 293.95291  | 31               |

MCC is Matthew's correlation coefficient

5fold cross-validation

| Method             | mAP      | 
|--------------------|:--------:|
|Linear regression   |          | 
|Logistic regression | 0.740    | 
|Decision trees      | 0.441    | 
|k Nearest Neighbors | 0.685    | 
|Random forest       | 0.774    | 
|linear SVM          | 0.656    |
|rbf SVM             | 0.699    |

