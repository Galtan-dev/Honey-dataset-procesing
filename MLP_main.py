import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Normalization
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# load data
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")

# one hot  encoding
one_hot_encoded = pd.get_dummies(data['Pollen_analysis'], prefix='Pollen_analysis')
data = pd.concat([data, one_hot_encoded], axis=1)

# standardization and normalizatiom
columns_to_scale = [col for col in data.columns if col != 'Pollen_analysis']
scaler = MinMaxScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
scaler = StandardScaler()
data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
data.drop(['Pollen_analysis'], axis=1, inplace=True)

# data separation
X = data.drop(columns=['Price', "EC", "F", "G", "pH", "Density", "Viscosity", "CS", "WC"])
y = data['Price']

# neuron site model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = Sequential([
        Normalization(axis=-1),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)])

model.compile(optimizer=Adam(0.001), loss="mean_absolute_error")
history = model.fit(x_train, y_train, epochs=15, batch_size=20, validation_split=0.3)

# test data prediction
predictions = model.predict(x_test)

# grafical resolutions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='prediction vs. actual plot')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='ideality')
plt.xlabel('Real values')
plt.ylabel('Prediceted values')
plt.legend()
plt.show()

# resolution parameters
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))