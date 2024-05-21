import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# load data and preprocesing
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")
data = data.sample(n=55000, random_state=42)

# one hot  encoding
one_hot_encoded = pd.get_dummies(data['Pollen_analysis'], prefix='Pollen_analysis')
data = pd.concat([data, one_hot_encoded], axis=1)

columns_to_scale = [col for col in data.columns if col != 'Pollen_analysis']
scaler = MinMaxScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data.drop(['Pollen_analysis'], axis=1, inplace=True)

# data separation
X = data.drop(columns=['Price', "EC", "F", "G", "pH", "Density", "Viscosity", "CS", "WC"])
y = data['Price']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# SVM for regresion fit
regre = svm.SVR(kernel="linear", epsilon=0.05, C=4, degree=3)
regre.fit(x_train, y_train)

# resolution parametrs
y_pred_train = regre.predict(x_train)
y_pred_test = regre.predict(x_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred_test))
print("R2 Score:", r2_score(y_test, y_pred_test))

# grafical resolution
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_test, label='prediction vs. actual plot')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='ideality')
plt.xlabel('Real values')
plt.ylabel('Predicted values')
plt.show()
