# Wine quality prediction

The data contains 1599 red wine quality parameters (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) together with quality label.

## The first experiment
Train/test split is done with ratio equal to 0.2 and with the shuffle equal to False.

Total 
3 occurs 10 times 
4 occurs 53 times 
5 occurs 681 times 
6 occurs 638 times 
7 occurs 199 times 
8 occurs 18 times 

Train 
3 occurs 5 times 
4 occurs 42 times 
5 occurs 538 times 
6 occurs 502 times 
7 occurs 177 times 
8 occurs 15 times 

Test 
3 occurs 5 times 
4 occurs 11 times 
5 occurs 143 times 
6 occurs 136 times 
7 occurs 22 times 
8 occurs 3 times 
8 occurs 3 times 

All quality parameters are used with different classifiers:
Results for run with non-shuffled train/test split and standardized input data:
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.359    | 0.384     | 0.181  | 0.191    | 0.195 | 205              |
|Logistic regression | 0.628    | 0.282     | 0.266  | 0.265    | 0.365 | 119              |
|Decision trees      | 0.522    | 0.249     | 0.271  | 0.255    | 0.237 | 153              |
|Gaussian Navie Bayes| 0.594    | 0.301     | 0.306  | 0.300    | 0.340 | 130              |
|k Nearest Neighbors | 0.622    | 0.308     | 0.294  | 0.296    | 0.384 | 158              |
|Random forest       | 0.597    | 0.290     | 0.292  | 0.290    | 0.317 | 129              |
|linear SVM          | 0.625    | 0.293     | 0.264  | 0.265    | 0.350 | 120              |
|rbf SVM             | 0.622    | 0.293     | 0.282  | 0.284    | 0.351 | 121              |


## The second experiment
Train/test split is done with ratio equal to 0.2 and with the stratification regarding to "quality".

Total 
3 occurs 10 times 
4 occurs 53 times 
5 occurs 681 times 
6 occurs 638 times 
7 occurs 199 times 
8 occurs 18 times 

Train 
3 occurs 8 times 
4 occurs 42 times 
5 occurs 545 times 
6 occurs 510 times 
7 occurs 159 times 
8 occurs 15 times 

Test 
3 occurs 2 times 
4 occurs 11 times 
5 occurs 136 times 
6 occurs 128 times 
7 occurs 40 times 
8 occurs 3 times 

All quality parameters are used with different classifiers:
Results for run with non-shuffled train/test split and standardized input data:
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.397    | 0.202     | 0.158  | 0.176    | 0.210 | 193              |
|Logistic regression | 0.628    | 0.311     | 0.294  | 0.296    | 0.392 | 119              |
|Decision trees      | 0.628    | 0.355     | 0.388  | 0.363    | 0.431 | 119              |
|Gaussian Navie Bayes| 0.569    | 0.310     | 0.343  | 0.319    | 0.358 | 138              |
|k Nearest Neighbors | 0.594    | 0.308     | 0.286  | 0.292    | 0.346 | 130              |
|Random forest       | 0.728    | 0.362     | 0.358  | 0.358    | 0.562 | 87               |
|linear SVM          | 0.606    | 0.201     | 0.244  | 0.220    | 0.341 | 126              |
|rbf SVM             | 0.656    | 0.324     | 0.305  | 0.307    | 0.438 | 110              |

With stratification, the results can be different with repeated runs.
We can notice that accuracy can be improved with "more fair" train/test split.
