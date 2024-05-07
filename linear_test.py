from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, normalize
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# load and preprocesing
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")
# one hot  encoding
one_hot_encoded = pd.get_dummies(data['Pollen_analysis'], prefix='Pollen_analysis')
data = pd.concat([data, one_hot_encoded], axis=1)

# standardizace a normalizace
columns_to_scale = [col for col in data.columns if col != 'Pollen_analysis']
scaler = MinMaxScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data.drop(['Pollen_analysis'], axis=1, inplace=True)

# print(data.head())
# print(data.info())

X = data.drop(columns=['Price', "EC", "F", "G"])
y = data['Price']

# Rozdělení dat na trénovací a testovací množinu
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializace a trénování lineárního regresního modelu
model = LinearRegression()
model.fit(X_train, y_train)

# Výpis koeficientů a interceptu
# print("Koeficient:", model.coef_)
# print("Intercept:", model.intercept_)
#
# Predikce na testovacích datech
predictions = model.predict(X_test)

# Výpočet a výpis MSE (Mean Squared Error)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
print('Model Accuracy Train :', r2_score(y_train,y_pred_train)*100,'%')
print('Model Accuracy Test  :', r2_score(y_test,y_pred_test)*100,'%')

# Vykreslení grafu
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='prediction vs. actual plot')  # Skutečné hodnoty vs. Predikce modelu
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='ideality')  # Diagonální čára pro porovnání
plt.xlabel('Real values')
plt.ylabel('Predicted values')
# plt.title('Skutečné hodnoty vs. Predikce modelu')
plt.show()

print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))