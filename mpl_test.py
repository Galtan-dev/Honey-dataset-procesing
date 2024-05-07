import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from sklearn.preprocessing import StandardScaler


# data loading and preparation
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")[["Price", "Pollen_analysis", "Purity", "pH", "Density"]]
feature_categorical = [cname for cname in data.columns if data[cname].dtype == "object"]
le = LabelEncoder()
for columna in feature_categorical:
    data[columna] = le.fit_transform(data[columna])


# input data visualization
# fig, ax = plt.subplots(1, 1, figsize=(6, 6))
# ax.scatter(data["Price"], data["Purity"], c=data["Pollen_analysis"])
# plt.show()

# normalization
scaler = MinMaxScaler()
data[['Price', 'pH', 'Density', 'Pollen_analysis']] = scaler.fit_transform(data[['Price', 'pH', 'Density', 'Pollen_analysis']])

# input and target declaration
X = data[['Price']]     #, "pH", "Density"  "Price"
y = data["Purity"]
# y = to_categorical(y, num_classes=1)  # One-hot encoding

# neuron model
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
model = Sequential([
    Input(shape=(1,)),
    Dense(8, activation="relu"),  #kernel_regularizer=l2(0.01)
    Dense(24, activation="relu"),
    Dense(68, activation="relu"),
    Dense(132, activation="relu"),
    Dense(264, activation="relu"),
    Dense(528, activation="relu"),
    Dense(1056, activation="relu"),
    Dense(1, activation="linear"),
])

opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=MeanSquaredLogarithmicError(), metrics=["accuracy"])
history = model.fit(x_train, y_train, epochs=5, batch_size=20, validation_split=0.3)


# Vykreslení loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Vykreslení accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Predikce na testovacích datech
predictions = model.predict(x_test)

# Vykreslení grafu
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Skutečné hodnoty')
plt.ylabel('Predikce modelu')
plt.title('Skutečné hodnoty vs. Predikce modelu')
plt.show()