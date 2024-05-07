import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from keras.layers import Normalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



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

X = data.drop(columns=['Price', "EC", "F", "G", "pH", "Density", "Viscosity", "CS", "WC"])
y = data['Price']



# neuron model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = Sequential([
        Normalization(axis=-1),
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)
])

# opt = Adam(learning_rate=0.1)
model.compile(optimizer=Adam(0.001), loss="mean_absolute_error")        #, metrics=["accuracy" 'mean_squared_error'  MeanSquaredLogarithmicError()
history = model.fit(x_train, y_train, epochs=15, batch_size=20, validation_split=0.3)


# Vykreslení loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
#
# # Vykreslení accuracy
# plt.plot(history.history['accuracy'], label='Train Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Model Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# Predikce na testovacích datech
predictions = model.predict(x_test)

# Vykreslení grafu
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='prediction vs. actual plot')  # Skutečné hodnoty vs. Predikce modelu
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4, label='ideality')  # Diagonální čára pro porovnání
plt.xlabel('Real values')
plt.ylabel('Prediceted values')
# plt.title('prediction vs. actual plot')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(x_test.iloc[:, 0], y_test, label='Skutečné hodnoty', color='blue')  # Zobrazení skutečných hodnot
plt.scatter(x_test.iloc[:, 0], predictions, label='Predikce modelu', color='red')  # Zobrazení predikcí modelu
plt.xlabel('Vstupní hodnoty')
plt.ylabel('Cílové hodnoty / Predikce modelu')
plt.title('Cílové hodnoty vs. Predikce modelu')
plt.legend()
plt.show()


print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("Mean Absolute Error:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))