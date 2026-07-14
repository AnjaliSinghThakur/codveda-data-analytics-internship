"""
Codveda Data Analytics Internship - Level 2 (Intermediate)
Task 3: Clustering Analysis (K-Means)
Dataset: iris.csv
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RAW_PATH = "/home/claude/dataset1/Data Set For Task/1) iris.csv"
PLOT_DIR = "/home/claude/work/plots"

df = pd.read_csv(RAW_PATH).drop_duplicates()  # dedupe (3 dupes identified earlier)
print("Shape after dedupe:", df.shape)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]

# 1. Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Determine optimal number of clusters using the elbow method
inertias = []
sil_scores = []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(7, 5))
plt.plot(list(K_range), inertias, marker="o", color="#2E86AB")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Within-Cluster SS)")
plt.title("Elbow Method for Optimal k")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l2_t3_elbow.png", dpi=120)
plt.close()

print("\nInertia by k:", dict(zip(K_range, [round(i, 2) for i in inertias])))
print("Silhouette score by k:", dict(zip(K_range, [round(s, 3) for s in sil_scores])))

# Elbow + silhouette both point to k=3 (matches the known 3 iris species)
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["cluster"] = kmeans_final.fit_predict(X_scaled)

print(f"\nUsing k={optimal_k} (matches 3 known iris species)")
print("\nCluster vs actual species cross-tab:")
print(pd.crosstab(df["cluster"], df["species"]))

# 3. Visualize clusters using 2D scatter plots (PCA reduced + raw feature pair)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["pc1"], df["pc2"] = pcs[:, 0], pcs[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.scatterplot(data=df, x="pc1", y="pc2", hue="cluster", palette="Set2", ax=axes[0])
axes[0].set_title(f"K-Means Clusters (k={optimal_k}) - PCA projection")

sns.scatterplot(data=df, x="petal_length", y="petal_width", hue="cluster", palette="Set2", ax=axes[1])
axes[1].set_title("K-Means Clusters - Petal Length vs Width")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l2_t3_clusters.png", dpi=120)
plt.close()

print("\nClustering analysis complete. Plots saved to", PLOT_DIR)
