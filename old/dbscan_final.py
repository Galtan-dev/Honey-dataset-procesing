import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, homogeneity_completeness_v_measure
from sklearn.preprocessing import LabelEncoder
import seaborn as sns

# Načtení dat
data = pd.read_csv('honey_purity_dataset.csv') 
data = data.sample(n=200000, random_state=42)

# Příprava dat
X = data[['Purity', 'Price', 'Pollen_analysis']]  # Příznaky
label_encoder = LabelEncoder()
X['Pollen_analysis'] = label_encoder.fit_transform(X['Pollen_analysis'])
X['Pollen_analysis'] += 2

# LOGARITMUS
X = np.log(X)
fig = plt.figure()
ax3d = plt.subplot(111, projection='3d')
scatter = ax3d.scatter(X['Purity'], X['Price'], X['Pollen_analysis'])
plt.show()

# Inicializace a trénování modelu DBSCAN
eps = 0.15
min_samples = 300
dbscan = DBSCAN(eps=eps, min_samples=min_samples)  
dbscan.fit(X)

# Vizualizace výsledků clustrování s DBSCAN
fig = plt.figure(figsize=(10, 8))
ax3d = fig.add_subplot(111, projection='3d')
scatter = ax3d.scatter(X['Purity'], X['Price'], X['Pollen_analysis'], c=dbscan.labels_, cmap='viridis', alpha=0.5)
legend = ax3d.legend(*scatter.legend_elements(), title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3d.set_xlabel('Log(Purity)')  
ax3d.set_ylabel('Log(Price)')   
ax3d.set_zlabel('Encoded Pollen_analysis')  
ax3d.set_title('DBSCAN Clustering Results')
plt.show()


# Metriky modelu:
# Silhouette Score
silhouette = silhouette_score(X, dbscan.labels_)

# Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(X, dbscan.labels_)

# Adjusted Rand Index 
ari = adjusted_rand_score(data['Pollen_analysis'], dbscan.labels_)

# Homogeneity, Completeness, V-measure 
h, c, v = homogeneity_completeness_v_measure(data['Pollen_analysis'], dbscan.labels_)

print(f"eps: {eps}, min_samples. {min_samples}")
print("Silhouette Score:", silhouette)
print("Davies-Bouldin Index:", davies_bouldin)
print("Adjusted Rand Index (ARI):", ari)
print("Homogeneity:", h)
print("Completeness:", c)
print("V-measure:", v)


# Vykreslení heatmapy hustoty
plt.figure(figsize=(8, 6))
sns.kdeplot(data=X, x='Purity', y='Price', fill=True, cmap='viridis', alpha=0.5)
plt.scatter(X['Purity'], X['Price'], c=dbscan.labels_, cmap='viridis', alpha=0.5)
plt.xlabel('Purity')
plt.ylabel('Price')
plt.title('Density Heatmap with DBSCAN Clustering')
plt.colorbar(label='Cluster')
plt.show()
