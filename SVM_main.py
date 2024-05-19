import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
# load data and preprocesing
data = pd.read_csv("honey_purity_dataset.csv")
data = data.sample(n=50000, random_state=42)
# one hot  encoding
one_hot_encoded = pd.get_dummies(data['Pollen_analysis'], prefix='Pollen_analysis')
data = pd.concat([data, one_hot_encoded], axis=1)

# standardizace a normalizace
columns_to_scale = [col for col in data.columns if col != 'Pollen_analysis']
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data.drop(['Pollen_analysis'], axis=1, inplace=True)

# data separation
X = data.drop(columns=['Price'])
y = data['Price']
print(X.head())
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# X = data.drop(columns=['Price'])
# y = data['Price']
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
# kernels = ["poly", "rbf", "sigmoid"]
# C_params = np.array([0.1, 1, 5, 10, 20, 100, 500, 1000, 5000, 10000])
# degrees = np.array([1,2,3,4,5])
# # table for results
# df_accuracy = pd.DataFrame({"kernel":[], "C":[],"degree": [], "fit_mse": [],
#                             "validation_mse":[]})
# # Trying different combinations
# for kernel in kernels:
#   for C in C_params:
#     for degree in degrees:
#       regre = svm.SVR(kernel=kernel, C=C)
#       regre.fit(x_train, y_train)
#       y_pred_train = regre.predict(x_train)
#       y_pred_test = regre.predict(x_test)
#       mse_fit = mean_squared_error(y_pred_train, y_train)
#       mse_val = mean_squared_error(y_pred_test, y_test)
#       print(f"Kernel: {kernel}. Regularization parameter C: {C:.0e}. degree: {degree}")
#       print(f"\t - Training error (MSE): {mse_fit:.2f}."
#           f" Validation error(MSE): {mse_val:.2f}.")
#       df_accuracy.loc[df_accuracy.shape[0]] = [kernel, C, degree, mse_fit, mse_val]

# # Finding the best combination of kernel and regularization
# best_regr = df_accuracy[df_accuracy.validation_mse == df_accuracy.validation_mse.min()].iloc[0,:]
# print(best_regr)

# SVM for regresion fit
regre = svm.SVR(kernel="rbf", C=10, degree=1)
regre.fit(x_train, y_train)

# resolution parametrs
y_pred_train = regre.predict(x_train)
y_pred_test = regre.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_test))
print("R2 Score:", r2_score(y_test, y_pred_test))  # no use

# grafical resolution
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, label='prediction vs. actual plot')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=4, label='Reference Line')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.legend()
plt.show()