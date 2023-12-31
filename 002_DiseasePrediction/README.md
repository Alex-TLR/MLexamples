# Disease prediction

The data contains 132 columns with indicated symptoms and one column with the prognosis. The symptom columns have numerical values 0 or 1, while prognosis gives a description of the prognosis.

There are 41 possible different prognosis, therefore there are N = 41 different possible outputs.

One Hot encoder encodes each output value with an array of N elements with one element equal to 1 and all other elements equal to 0. The position of the element equal to 1 indicates the corresponding label.

Ordinary encoder assigns different numerical values to each different category.

For example:

|   | Prognosis                               | One Hot encoder                                                                   | Ordinary encoder | 
|---|:---------------------------------------:|:----------------------------------------------------------------------------------|:----------------:|
|0  | Fungal infection                        | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 15               |
|1  | Allergy                                 | 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 4                |
|2  | GERD                                    | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 16               |
|3  | Chronic cholestasis                     | 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 9                |
|4  | Drug Reaction                           | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 14               |
|5  | Peptic ulcer diseae                     | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 | 33               |
|6  | AIDS                                    | 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 1                |
|7  | Diabetes                                | 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 12               |
|8  | Gastroenteritis                         | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 17               |
|9  | Bronchial Asthma                        | 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 6                |
|10 | Hypertension                            | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 23               |
|11 | Migraine                                | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 | 30               |
|12 | Cervical spondylosis                    | 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 7                |
|13 | Paralysis (brain hemorrhage)            | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 | 32               |
|14 | Jaundice                                | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 | 28               |
|15 | Malaria                                 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 | 29               |
|16 | Chicken pox                             | 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 8                |
|17 | Dengue                                  | 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 11               |
|18 | Typhoid                                 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 | 37               |
|19 | hepatitis A                             | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 | 40               |
|20 | Hepatitis B                             | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 19               |
|21 | Hepatitis C                             | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 20               |
|22 | Hepatitis D                             | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 21               |
|23 | Hepatitis E                             | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 22               |
|24 | Alcoholic hepatitis                     | 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 3                |
|25 | Tuberculosis                            | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 | 36               |
|26 | Common Cold                             | 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 10               |
|27 | Pneumonia                               | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 | 34               |
|28 | Dimorphic hemmorhoids(piles)            | 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 13               |
|29 | Heart attack                            | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 18               |
|30 | Varicose veins                          | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 | 39               |
|31 | Hypothyroidism                          | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 26               |
|32 | Hyperthyroidism                         | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 24               |
|33 | Hypoglycemia                            | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 25               |
|34 | Osteoarthristis                         | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 | 31               |
|35 | Arthritis                               | 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 5                |
|36 | Paroymsal Positional Vertigo            | 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 0                |
|37 | Acne                                    | 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 | 2                |
|38 | Urinary tract infection                 | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 | 38               |
|39 | Psoriasis                               | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 | 35               |
|40 | Impetigo                                | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 | 27               |


## The first experiment
In the first experiment, we used train/test split as defined in Training.csv and Testing.csv files. The Testing.csv contains only 42 testing cases, therefore it is not descriptive enough.
Every kind of classifier will probably have the best possible performance.

Results :
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.976    | 0.988     | 0.988  | 0.984    | 0.976 | 1                |
|Logistic regression | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|Decision trees      | 0.976    | 0.988     | 0.988  | 0.984    | 0.976 | 1                |
|Gaussian Naive Bayes| 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|k Nearest Neighbors | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|Random forest       | 0.976    | 0.988     | 0.988  | 0.984    | 0.976 | 1                |
|linear SVM          | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |

## The second experiment
Both data from Training.csv and Testing.csv files are concatenated, afterwitch the data is split into train/test sets.
In this experiment train and test are split in 50% manner.
Except Linear Regression model, all other classifiers has near-perfect or perfect performance.  
Number of both train and test samples is equal to 2481.

Results :
| Method             | Accuracy | Precision | Recall | F1-Score | MCC   | Number of errors |
|--------------------|:--------:|:---------:|:------:|:--------:|:-----:|:----------------:|
|Linear regression   | 0.989    | 0.990     | 0.989  | 0.989    | 0.988 | 28               |
|Logistic regression | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|Decision trees      | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|Gaussian Naive Bayes| 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|k Nearest Neighbors | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
|Random forest       | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 1                |
|linear SVM          | 1.000    | 1.000     | 1.000  | 1.000    | 1.000 | 0                |
