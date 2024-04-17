import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# data loading
df = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")

# erasing of non use column
df = df.drop("Pollen_analysis", axis=1)
# statistic print
print(df.info())

# corelation matrix
ax = plt.subplot(111)
sns.heatmap(df.corr(), annot=True, ax=ax)
plt.savefig("correl_matrix.png")
plt.show()