import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# data loading
df = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")

# erasing of non use column
df = df.drop("Pollen_analysis", axis=1)
# statistic print
print(df.info())
print(df)

x = np.array(df[df.columns[8]])
y = np.array(df[df.columns[9]])
z = np.array(df[df.columns[1]])

# plt.scatter(x, y, z)
# plt.xlabel("purity")
# plt.ylabel("price")
# plt.zlabel("pH")
# plt.show()

fig = plt.figure()
ax3d = plt.subplot(111, projection="3d")
ax3d.scatter(x, y, z)
ax3d.set_xlabel("purity")
ax3d.set_ylabel("price")
ax3d.set_zlabel("pH")
plt.show()


#
# # corelation matrix
# ax = plt.subplot(111)
# sns.heatmap(df.corr(), annot=True, ax=ax)
# plt.savefig("correl_matrix.png")
# plt.show()