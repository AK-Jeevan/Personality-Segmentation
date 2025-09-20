# This Model will segment individuals based on their personality traits using DBSCAN Clustering and PCA for dimensionality reduction.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Load the dataset
data = pd.read_csv(r"C:\Users\akjee\Documents\AI\ML\Unsupervised Learning\personality_synthetic_dataset.csv")

# Step 2: Remove duplicates
data.drop_duplicates(inplace=True)

# Step 3: Separate personality types and keep only numeric columns
personality_types = data['personality_type']
X = data.drop(columns=['personality_type'])
X = X.select_dtypes(include=[np.number])  # Keep only numeric features
X = X.dropna()  # Remove rows with missing values
personality_types = personality_types.loc[X.index]  # Align personality types with cleaned data

# Step 4: Check feature variance (for troubleshooting)
print("Feature variance after cleaning:")
print(X.var())
print("\nFeature description:")
print(X.describe())

# Step 5: Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Reduce dimensions to 2D for visualization
pca = PCA(n_components=2, random_state=0)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Apply DBSCAN clustering (already done)
dbscan = DBSCAN(eps=3.0, min_samples=4)
labels = dbscan.fit_predict(X_scaled)

# Step 7: Calculate silhouette score (excluding noise points)
mask = labels != -1
if np.unique(labels[mask]).size > 1:
    sil_score = silhouette_score(X_scaled[mask], labels[mask])
    print(f"Silhouette Score (excluding noise): {sil_score:.3f}")
else:
    print("Not enough clusters for silhouette score.")

# Step 8: Visualize clusters in PCA space
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=50)
plt.title('DBSCAN Clusters (PCA 2D)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Step 9: Print cluster counts and outliers
print("Cluster counts (including noise = -1):")
print(pd.Series(labels).value_counts().sort_index())
print("\nNumber of outliers:", np.sum(labels == -1))