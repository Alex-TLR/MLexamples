# Wine quality prediction

The data contains 1599 red wine quality parameters (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) and quality labels.

## The first experiment
Train/test split is done with a ratio of 0.2 and the shuffle equal to False.

Total 
3 occurs 10 times, 
4 occurs 53 times, 
5 occurs 681 times, 
6 occurs 638 times, 
7 occurs 199 times, 
8 occurs 18 times. 

Train 
3 occurs 5 times, 
4 occurs 42 times, 
5 occurs 538 times, 
6 occurs 502 times, 
7 occurs 177 times, 
8 occurs 15 times. 

Test (total 319)
3 occurs 5 times, 
4 occurs 11 times, 
5 occurs 143 times, 
6 occurs 136 times, 
7 occurs 22 times, 
8 occurs 3 times.  

All classifiers are used with default parameters.
Results for run with non-shuffled train/test split and standardized input data:
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.359    | 0.384     | 0.181  | 0.191    | 0.195 | 205              |
|Logistic regression | 0.628    | 0.282     | 0.266  | 0.265    | 0.365 | 119              |
|Decision trees      | 0.494    | 0.264     | 0.288  | 0.268    | 0.209 | 162              |
|Gaussian Navie Bayes| 0.594    | 0.301     | 0.306  | 0.300    | 0.340 | 130              |
|k Nearest Neighbors | 0.506    | 0.247     | 0.252  | 0.245    | 0.194 | 158              |
|Random forest       | 0.609    | 0.278     | 0.284  | 0.280    | 0.337 | 125              |
|linear SVM          | 0.625    | 0.293     | 0.264  | 0.265    | 0.350 | 120              |
|rbf SVM             | 0.622    | 0.293     | 0.282  | 0.284    | 0.351 | 121              |

MCC is Matthew's correlation coefficient. 
Overall performance is weak. Reasons for this could be found in small number of examples of very low and very 
high quality wines, or in highly correlated features. For example, the correlation between 'fixed acidity' and
'critic acid' is 0.67. 

Finally, we can compare the accuracy with original train/test split and stratified split (according to 'quality')

Train
3 occurs 8 times,  
4 occurs 42 times,  
5 occurs 545 times,  
6 occurs 510 times,  
7 occurs 159 times,  
8 occurs 15 times.  

Test  
3 occurs 2 times,  
4 occurs 11 times,  
5 occurs 136 times,  
6 occurs 128 times,  
7 occurs 40 times,  
8 occurs 3 times.  

Stratification accoriding the 'quality' forces the 0.2 split so the each category approximates that ratio. 

|                    |: non-shuffled              :|: stratified                :| 
| Method             | Accuracy | Number of errors | Accuracy | Number of errors |
|--------------------|:--------:|:----------------:|:--------:|:----------------:|
|Linear regression   | 0.359    | 205              | 0.371    | 201              |
|Logistic regression | 0.628    | 119              | 0.628    | 119              |
|Decision trees      | 0.494    | 162              | 0.613    | 124              |
|Gaussian Navie Bayes| 0.594    | 130              | 0.609    | 125              |
|k Nearest Neighbors | 0.506    | 158              | 0.631    | 118              |
|Random forest       | 0.609    | 125              | 0.706    | 94               |
|linear SVM          | 0.625    | 120              | 0.603    | 127              |
|rbf SVM             | 0.622    | 121              | 0.662    | 108              |


## The second experiment
Train/test split is done with a ratio of 0.2 and the shuffle equal to False.
Categories (wine qualities) with the small value of samples are discarded in this example.
The classification is done for quility values of 5, 6 and 7.

Total 
3 occurs 10 times,
4 occurs 53 times, 
5 occurs 681 times, 
6 occurs 638 times, 
7 occurs 199 times, 
8 occurs 18 times. 

Train  
5 occurs 538 times,  
6 occurs 499 times,  
7 occurs 177 times.  

Test (total 304)
5 occurs 143 times,  
6 occurs 139 times,  
7 occurs 22 times.  

All classifiers are used with default parameters.
Results for run with non-shuffled train/test split and standardized input data:
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |  
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.597    | 0.391     | 0.439  | 0.430    | 0.248 | 128              | 
|Logistic regression | 0.655    | 0.583     | 0.522  | 0.535    | 0.378 | 105              | 
|Decision trees      | 0.569    | 0.499     | 0.550  | 0.506    | 0.285 | 131              | 
|Gaussian Navie Bayes| 0.641    | 0.561     | 0.564  | 0.560    | 0.371 | 109              | 
|k Nearest Neighbors | 0.543    | 0.466     | 0.480  | 0.467    | 0.211 | 139              | 
|Random forest       | 0.641    | 0.587     | 0.564  | 0.573    | 0.362 | 109              | 
|linear SVM          | 0.658    | 0.592     | 0.524  | 0.539    | 0.377 | 104              | 
|rbf SVM             | 0.655    | 0.609     | 0.560  | 0.577    | 0.378 | 105              |

MCC is Matthew's correlation coefficient. 
There is notably best accuracy comparing to the results from the first experiment. However, considering that 
there is less test samples in this case, the improvement is not significant.
However, if stratified train/test split is used Random forest classifier accuracy is 0.760 with 73 number of errors.

## The third experiment
In this experiment, we are analysing the correlation between the different features.
'Fixed acidity' is highly correlated with 'Critic acid' and 'Density', also 'Free sulfur dioxide' is highly correlated with 
'Total sulfur dioxide'.
Therefore, we will remove 'Fixed acidity' and 'Free sulfur dioxide'.
In this experiment we combined the approaches from the first two experiments: the lowest and the highest qualities are removed, and also stratified split is used.

Results for run with stratified train/test split and standardized input data:
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |  
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.559    | 0.364     | 0.423  | 0.374    | 0.241 | 134              | 
|Logistic regression | 0.599    | 0.580     | 0.517  | 0.531    | 0.317 | 122              | 
|Decision trees      | 0.618    | 0.566     | 0.555  | 0.559    | 0.363 | 116              | 
|Gaussian Navie Bayes| 0.553    | 0.527     | 0.550  | 0.536    | 0.275 | 136              | 
|k Nearest Neighbors | 0.595    | 0.551     | 0.526  | 0.535    | 0.319 | 123              | 
|Random forest       | 0.684    | 0.668     | 0.629  | 0.644    | 0.471 | 96               | 
|linear SVM          | 0.582    | 0.394     | 0.447  | 0.417    | 0.273 | 127              | 
|rbf SVM             | 0.589    | 0.551     | 0.491  | 0.498    | 0.295 | 125              |

