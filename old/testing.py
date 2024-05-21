import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# predefining functions
def mape(data_y, data_x, theta):
    return 1 / m * np.sum(np.abs(data_y - (theta[0] * data_x[:, 0] + theta[1])) / data_y) * 100

def mse(data_y, data_x, theta):
    return 1 / m * np.sum((data_y - (theta[0] * data_x[:, 0] + theta[1])) ** 2)

def linear_function(data_x, theta):
    return theta[0] * data_x[:, 0] + theta[1]

# selection = [1, 3, 9]
selection = [9]
# data loading and preparation
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")
data = data.drop("Pollen_analysis", axis=1)
data = data.iloc[:10000, :]
X = np.array(data[data.columns[selection]])
y = np.array(data[data.columns[8]])

# MinMax Scaler
minmax = MinMaxScaler((1, 2))
data_scaled = minmax.fit_transform(data)
x_scaled = data_scaled[:, [1]]
y_scaled = data_scaled[:, 8]

print("Prep complete")

## inicializace parametrů
# Learning coefficient, maximum epochs limit, error evaluation
alpha = 4e-10
epochs_max = 10
err_min = 0.022
# Number of features
n = len(selection)
# Length of dataset
m = len(y)
# Random order of the data
indices = np.linspace(0, m - 1, m, dtype=np.dtype("int"))
np.random.shuffle(indices)
# Declaring loss function and linear parameters
theta = np.zeros(n + 1)
grad_theta = np.zeros(n + 1)
loss = [mape(y, X, theta)]
print("Param inic complete")

## stochastic gradient descent
epoch = 0
while epoch < epochs_max:
    epoch += 1
    for i in indices:
        grad_common = - 2 * (y[i] - (theta[0] * X[i, 0] + theta[1]))
        # print(f"grad_c:{grad_common}")
        grad_theta[0] = grad_common * X[i, 0]
        grad_theta[1] = grad_common * 1

        theta[0] = theta[0] - alpha * grad_theta[0]
        theta[1] = theta[1] - alpha * grad_theta[1]
        # print(f"grad:{grad_theta[0]}")

        # print(f"y: {y}, x:{X}, theta:{theta}")
        loss.append(mse(y, X, theta))
        print(f"loss: {loss[-1]}, err: {err_min}")
        if loss[-1] <= err_min:
            print("Sufficient error reached, finishing the calculation.")
            break
    else:
        print(f"loss: {loss[-1]}, err: {err_min}")
        continue
    break

print("loop complete")

fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(X[:, 0], y, label='Actual data')
ax.scatter(X[:, 0], linear_function(X, theta), color='r', label='Predicted data')

# Line to visualize the model
line_x = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
line_y = theta[0] * line_x + theta[1]
ax.plot(line_x, line_y, color="crimson", label="Model")

ax.set_xlabel("Feature")
ax.set_ylabel("Target")
ax.legend()
plt.show()

fig_error = plt.figure(figsize=(12, 6))
plt.subplots_adjust(top=0.9, bottom=0.075, left=0.065, right=0.975, hspace=0.2,
                    wspace=0.155)
fig_error.suptitle("Modelling Accuracy", fontweight="bold")
ax_error_rel = plt.subplot(211)
h = linear_function(X, theta)
ax_error_rel.scatter(np.linspace(0, m - 1, m), (h - y) / y * 100, marker="*",
                     color="crimson", edgecolor="crimson")
ax_error_rel.set_xlabel("sample #")
ax_error_rel.set_ylabel("relative error (%)")

ax_loss = plt.subplot(212)
ax_loss.plot(np.linspace(0, len(loss), len(loss[::500])), loss[::500],
             color="crimson")
ax_loss.set_xlim(0, 5000)
ax_loss.set_xlabel("iteration #")
ax_loss.set_ylabel("total loss")
plt.show()
