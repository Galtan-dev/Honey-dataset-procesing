import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from category_encoders.target_encoder import TargetEncoder
from sklearn.preprocessing import StandardScaler

# load and preprocesing
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")
# one hot  encoding
# one_hot_encoded = pd.get_dummies(data['Pollen_analysis'], prefix='Pollen_analysis')
# data = pd.concat([data, one_hot_encoded], axis=1)
#
# # standardizace a normalizace
# columns_to_scale = [col for col in data.columns if col != 'Pollen_analysis']
# scaler = MinMaxScaler()
# data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
# scaler = StandardScaler()
# data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])
# data.drop(['Pollen_analysis'], axis=1, inplace=True)

labels_pollen = np.array(data[data.columns[7]])

# španělský label encoding
feature_categorical = [cname for cname in data.columns if data[cname].dtype == "object"]
le = LabelEncoder()
for columna in feature_categorical:
    data[columna] = le.fit_transform(data[columna])

print(data.info())
print(data.head())



# # Získání dat pro graf
# x = np.array(data[data.columns[9]])
# y = np.array(data[data.columns[10]])
# z = np.array(data[data.columns[7]])
# # Vytvoření grafu
# fig = plt.figure()
# ax3d = plt.subplot(111)
# scatter = ax3d.scatter(x, y, c=z, s=20)
# # Nastavení popisků os
# ax3d.set_xlabel("Purity")
# ax3d.set_ylabel("Price")
# # Vytvoření legendy
# # legend1 = ax3d.legend(*scatter.legend_elements(),
# #                     loc="upper right", title="Pollen Analysis")
# # ax3d.add_artist(legend1)
# # Zobrazení grafu
# plt.show()



x = np.array(data[data.columns[9]])
y = np.array(data[data.columns[7]])
z = np.array(data[data.columns[1]])
fig = plt.figure()
ax3d = plt.subplot(111) #projection="3d"
ax3d.scatter(x, y, c=z, s=2)
ax3d.set_xlabel("-")
ax3d.set_ylabel(".")
plt.show()

# # corelation matrix
# ax = plt.subplot(111)
# sns.heatmap(data.corr(), annot=True, ax=ax)
# plt.savefig("correl_matrix.png")
# plt.show()


# # koláčové grafy
# pollen = np.array(data[data.columns[7]])
# unique_values, counts = np.unique(pollen, return_counts=True)
# labeling = np.unique(labels_pollen)
# plt.figure(figsize=(8, 8))
# plt.pie(counts, labels=labeling, autopct='%1.1f%%', startangle=140)
# plt.axis('equal')
# # plt.title('Pollen analyses')
# plt.show()