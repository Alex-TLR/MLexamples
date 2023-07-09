from sklearn import linear_model
from sklearn.metrics import confusion_matrix
import printResults
from time import time

def linearRegression(x_train, x_test, y_train, y_test):

    # Make linear regression model
    start = time()
    LinearRegression = linear_model.LinearRegression()
    LinRegModel = LinearRegression.fit(x_train, y_train)
    
    # Predict 
    y_pred = LinRegModel.predict(x_test)
    y_pred = [0 if i <= 0.5 else 1 for i in y_pred]
    stop = time()
    print(f'Time spent is {stop - start} seconds.')

    # Print report
    printResults.printResults(y_test, y_pred, "Linear Regression")
    print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
    print("\n")
    return None 

    
# Number of misclassified cases
# N_errors  

# # Evaluate the model
# MSE = mean_squared_error(y_test, y_pred)
# RMSE = np.sqrt(MSE)
# MAE = mean_absolute_error(y_test, y_pred)
# MAPE = mean_absolute_percentage_error(y_test, y_pred)
# R2 = r2_score(y_test, y_pred)

# print("Linear regression model")
# print("MSE", MSE)
# print("RMES", RMSE)
# print("MAE", MAE)
# print("R2 score coefficient of determination", R2)

# # Plot Class (maybe) vs Amount for linear regression
# plt.figure
# plt.title("Class vs Amount")
# plt.xlabel("Amount")
# plt.ylabel("Class")
# plt.scatter(x_test[:, -1], y_test, c = 'b')
# plt.scatter(x_test[:, -1], y_pred, c = 'r')
# plt.show()


# # Make logistic regression model
# LogRegression = linear_model.LogisticRegression(multi_class='ovr')
# LogRegModel = LogRegression.fit(x_train, y_train)

# # Predict the test
# y_pred = LogRegModel.predict(x_test)

# # Evaluate the model
# MSE = mean_squared_error(y_test, y_pred)
# RMSE = np.sqrt(MSE)
# MAE = mean_absolute_error(y_test, y_pred)
# MAPE = mean_absolute_percentage_error(y_test, y_pred)
# R2 = r2_score(y_test, y_pred)

# print("Logistic regression model")
# print("MSE", MSE)
# print("RMES", RMSE)
# print("MAE", MAE)
# print("R2 score coefficient of determination", R2)


# # Plot Class (maybe) vs Amount for linear regression
# plt.figure
# plt.title("Class vs Amount")
# plt.xlabel("Amount")
# plt.ylabel("Class")
# plt.scatter(x_test[:, -1], y_test, c = 'b')
# plt.scatter(x_test[:, -1], y_pred, c = 'r')
# plt.show()


# plt.figure(figsize = (12,6))
# plt.subplots_adjust(hspace = 0.5)
# for i in range(2):
#     plt.subplot(121 + i)

