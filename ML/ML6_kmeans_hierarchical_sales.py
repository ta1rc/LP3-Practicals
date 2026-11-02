"""
===============================================================================
ML6 : Implement K-Means clustering / hierarchical clustering on
      sales_data_sample.csv dataset. Determine the number of clusters using
      the elbow method.
      Dataset: https://www.kaggle.com/datasets/kyanyoga/sample-sales-data
      (download "sales_data_sample.csv" and keep it in the same folder)
===============================================================================

THEORY
------
1. UNSUPERVISED LEARNING / CLUSTERING (Unit V)
   No target labels. CLUSTERING groups data so that points in the same
   cluster are similar (high intra-cluster similarity) and points in different
   clusters are dissimilar (low inter-cluster similarity).
   Applications: customer segmentation, market basket analysis, image
   segmentation, anomaly detection.

2. K-MEANS ALGORITHM (partitioning, centroid based)
       1. Choose k; initialise k centroids (randomly / k-means++).
       2. ASSIGN each point to the nearest centroid (Euclidean distance).
       3. UPDATE each centroid = mean of points assigned to it.
       4. Repeat 2-3 until centroids do not change (convergence).
   Objective: minimise WCSS / inertia = SUM_k SUM_{x in C_k} ||x - mu_k||^2.
   Time O(n k d t). Sensitive to initialisation, outliers and scale; assumes
   spherical clusters; k must be given.
   K-MEDOIDS (PAM): uses actual data points (medoids) as centres -> robust to
   outliers.

3. ELBOW METHOD
   Run K-means for k = 1..10 and plot WCSS vs k. WCSS always decreases as k
   increases; the point where the decrease sharply slows down ("elbow") is
   chosen as the optimal k. (Here also detected automatically using maximum
   distance from the line joining first and last points.)
   Other intrinsic measures: SILHOUETTE coefficient s = (b - a)/max(a, b),
   in [-1, 1] (a = mean intra-cluster distance, b = mean nearest-cluster
   distance); Davies-Bouldin index. EXTRINSIC measures (need ground truth):
   purity, Rand index, NMI.

4. HIERARCHICAL CLUSTERING
   Builds a tree of clusters (DENDROGRAM).
   - AGGLOMERATIVE (bottom-up): start with each point as a cluster, merge the
     two closest clusters repeatedly.
   - DIVISIVE (top-down): start with one cluster, split recursively.
   Linkage (distance between clusters): single (min), complete (max),
   average, WARD (minimum increase in variance).
   No need to fix k in advance - cut the dendrogram at a height. O(n^2) memory.

5. OTHER CLUSTERING (syllabus)
   - DBSCAN (density based): eps radius & minPts; finds arbitrary shapes and
     labels noise points; no k needed.
   - Spectral clustering: uses eigenvectors of a similarity graph Laplacian.

6. PRE-PROCESSING HERE
   Numeric features QUANTITYORDERED, PRICEEACH, SALES, MSRP (+ encoded
   DEALSIZE) are selected and STANDARDIZED because K-means is distance based.

CONCLUSION: The elbow curve shows the optimal number of clusters (about 3),
and both K-means and agglomerative clustering segment the sales orders into
groups such as small / medium / large deals.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "sales_data_sample.csv")

# %% 1. Read & pre-process ----------------------------------------------------
df = pd.read_csv(DATA, encoding="latin1")      # file is not UTF-8
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])

features = ["QUANTITYORDERED", "PRICEEACH", "SALES", "MSRP"]
data = df[features].copy()
if "DEALSIZE" in df.columns:
    data["DEALSIZE"] = df["DEALSIZE"].map({"Small": 0, "Medium": 1,
                                           "Large": 2})
data = data.dropna()
print("\nFeatures used:", list(data.columns))

X = StandardScaler().fit_transform(data)

# %% 2. Elbow method ----------------------------------------------------------
wcss = []
K = range(1, 11)
for k in K:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    km.fit(X)
    wcss.append(km.inertia_)
    print(f"k = {k:2d}  WCSS = {km.inertia_:.2f}")

# automatic elbow: point farthest from line joining first & last WCSS points
p1, p2 = np.array([1, wcss[0]]), np.array([10, wcss[-1]])
d = p2 - p1
dists = [abs(d[0] * (p1[1] - w) - d[1] * (p1[0] - k)) / np.linalg.norm(d)
         for k, w in zip(K, wcss)]
best_k = list(K)[int(np.argmax(dists))]
print("\nOptimal number of clusters (elbow) =", best_k)

plt.plot(list(K), wcss, marker="o")
plt.axvline(best_k, color="r", linestyle="--", label=f"elbow k={best_k}")
plt.xlabel("Number of clusters k")
plt.ylabel("WCSS (inertia)")
plt.title("Elbow method")
plt.legend()
plt.show()

# %% 3. K-Means with optimal k ------------------------------------------------
kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10,
                random_state=42)
data["KMeans_Cluster"] = kmeans.fit_predict(X)
print(f"\nK-Means silhouette score = "
      f"{silhouette_score(X, data['KMeans_Cluster']):.4f}")
print("Cluster sizes:\n", data["KMeans_Cluster"].value_counts().sort_index())
print("\nCluster profile (mean of features):")
print(data.groupby("KMeans_Cluster")[features].mean().round(2))

plt.scatter(data["QUANTITYORDERED"], data["SALES"],
            c=data["KMeans_Cluster"], cmap="viridis", s=10)
plt.xlabel("QUANTITYORDERED")
plt.ylabel("SALES")
plt.title(f"K-Means clusters (k={best_k})")
plt.show()

# %% 4. Hierarchical (agglomerative) clustering -------------------------------
sample = X[np.random.RandomState(42).choice(len(X), min(500, len(X)),
                                            replace=False)]
plt.figure(figsize=(12, 5))
dendrogram(linkage(sample, method="ward"), truncate_mode="level", p=5)
plt.title("Dendrogram (Ward linkage, 500-point sample)")
plt.xlabel("Samples")
plt.ylabel("Euclidean distance")
plt.show()

agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
data["Hier_Cluster"] = agg.fit_predict(X)
print(f"\nHierarchical silhouette score = "
      f"{silhouette_score(X, data['Hier_Cluster']):.4f}")
print("Cluster sizes:\n", data["Hier_Cluster"].value_counts().sort_index())

plt.scatter(data["QUANTITYORDERED"], data["SALES"],
            c=data["Hier_Cluster"], cmap="plasma", s=10)
plt.xlabel("QUANTITYORDERED")
plt.ylabel("SALES")
plt.title(f"Agglomerative clusters (k={best_k})")
plt.show()
