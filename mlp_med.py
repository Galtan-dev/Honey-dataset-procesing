import pandas as pd
from sklearn.preprocessing import LabelEncoder



# data loading and preparation
data = pd.read_csv("C:\\Users\\jakub\\PycharmProjects\\honey_ml\\data\\honey_purity_dataset.csv")[["Price", "Pollen_analysis", "Purity"]]
feature_categorical = [cname for cname in data.columns if data[cname].dtype == "object"]
le = LabelEncoder()
for columna in feature_categorical:
    data[columna] = le.fit_transform(data[columna])