import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder


# predefining functions
def mape(data_y, data_x, theta):
    return 1 / m * np.sum(np.abs(data_y - (theta[0] * data_x[:, 0] + theta[1] *
                        data_x[:, 1] + theta[2])) / data_y) * 100

def mse(data_y, data_x, theta):
    return 1 / m * np.sum((data_y - (theta[0] * data_x[:, 0] + theta[1] *
                        data_x[:, 1] + theta[2])) ** 2)

def linear_function(data_x, theta):
    return theta[0] * data_x[:, 0] + theta[1] * data_x[:, 1] + theta[2]

# selection = [1, 3, 9]
selection = [7, 9]
# data loading and preparation
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")

feature_categorical = [cname for cname in data.columns if data[cname].dtype == "object"]
le = LabelEncoder()
for columna in feature_categorical:
    data[columna] = le.fit_transform(data[columna])

# data = data.drop("Pollen_analysis", axis=1)
data = data.iloc[:10000, :]
X = np.array(data[data.columns[selection]])
y = np.array(data[data.columns[10]])

# MinMax Scaler
minmax = MinMaxScaler((1, 2))
data_scaled = minmax.fit_transform(data)
x_scaled = data_scaled[:, [7, 9]]
y_scaled = data_scaled[:, 10]

# Standardization
standardizer = StandardScaler()
data_standardized = standardizer.fit_transform(data)
x_standardized = data_standardized[:, [7, 9]]
y_standardized = data_standardized[:, 10]
print("Prep complete")


## inicializace parametrů
# Learning coefficient, maximum epochs limit, error evaluation
alpha = 4e-4
epochs_max = 10
err_min = 1
# Length and number of features
m, n = X.shape
# Random order of the data
indices = np.linspace(0, m - 1, m, dtype=np.dtype("int"))
np.random.shuffle(indices)
# Declaring loss function and linear parameters
# h(x) = theta0*x0 + theta1*x1 + theta2
theta = np.zeros(n + 1)
grad_theta = np.zeros(n + 1)
loss = [mape(y, X, theta)]
print("Param inic complete")


## stochastic gradient descent
epoch = 0
while epoch < epochs_max:
    epoch += 1
    for i in indices:
        grad_common = - 2 * (y[i] - (theta[0] * X[i, 0] + theta[1] * X[i, 1]
                                        + theta[2]))
        grad_theta[0] = grad_common * X[i, 0]
        grad_theta[1] = grad_common * X[i, 1]
        grad_theta[2] = grad_common * 1

        theta[0] = theta[0] - alpha * grad_theta[0]
        theta[1] = theta[1] - alpha * grad_theta[1]
        theta[2] = theta[2] - alpha * grad_theta[2]

        loss.append(mse(y, X, theta))
        print(f"loss: {loss[-1]}, err: {err_min}")
        if loss[-1] <= err_min:
            print("Sufficient error reached, finishing the calculation.")
            break
    else:
        print(f"loss: {loss[-1]}, err: {err_min}")
        continue  # only executed if the inner loop did NOT break
    break  # only executed if the inner loop DID break
print("loop complete")


fig = plt.figure()
ax3d = plt.subplot(111, projection="3d")
ax3d.scatter(X[:, 0], y)  # , X[:, 1]
# ax3d.set_title("Graf naměřených bodů", fontweight="bold")
# ax3d.set_xlabel("podíl sušiny (hm. %)")
# ax3d.set_ylabel("teplota (°C)")
# ax3d.set_zlabel("hustota (kg$\\cdot$m$^{-3}$)")

# Mesh to visualize the plane based on the model
mesh_x = np.linspace(min(X[:, 0]), max(X[:, 0]), 30)
mesh_y = np.linspace(min(X[:, 1]), max(X[:, 1]), 30)
mesh_x, mesh_y = np.meshgrid(mesh_x, mesh_y)
mesh_z = theta[0] * mesh_x + theta[1] * mesh_y + theta[2]
ax3d.plot_surface(mesh_x, mesh_y, mesh_z, color="crimson", alpha=0.5)
plt.show()


fig_error = plt.figure(figsize=(12, 6))
plt.subplots_adjust(top=0.9, bottom=0.075, left=0.065, right=0.975, hspace=0.2,
                    wspace=0.155)
fig_error.suptitle("Modelling Accuracy", fontweight="bold")
ax_error_rel = plt.subplot(221)
h = linear_function(X, theta)
ax_error_rel.scatter(np.linspace(0, m - 1, m), (h - y) / y * 100, marker="*",
                     color="crimson", edgecolor="crimson")
ax_error_rel.set_xlabel("sample #")
ax_error_rel.set_ylabel("residual error (%)")

ax_error_abs = plt.subplot(223)
h = linear_function(X, theta)
ax_error_abs.scatter(np.linspace(0, m - 1, m), (h - y), marker="*",
                     color="crimson", edgecolor="crimson")
ax_error_abs.set_xlabel("sample #")
ax_error_abs.set_ylabel("residual error")

ax_loss = plt.subplot(122)
ax_loss.plot(np.linspace(0, len(loss), len(loss[::500])), loss[::500],
             color="crimson")
ax_loss.set_xlabel("iteration #")
ax_loss.set_ylabel("total loss")
plt.show()


