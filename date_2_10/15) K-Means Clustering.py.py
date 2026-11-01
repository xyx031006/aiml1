#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

df = pd.read_csv("Mall_Customers.csv")

df.columns = df.columns.str.strip()

print("Dataset Information")
print(df.head())
print("\nDataset Shape:", df.shape)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

optimal_k = 5

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCustomer Cluster Analysis")
print(df.groupby("Cluster")[["Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2))

print("\nNumber of Customers in Each Cluster")
print(df["Cluster"].value_counts().sort_index())

score = silhouette_score(X_scaled, df["Cluster"])
print("\nSilhouette Score:", round(score, 4))

plt.figure(figsize=(9, 6))
plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=df["Cluster"],
    cmap="viridis",
    s=50
)

centers = kmeans.cluster_centers_
plt.scatter(
    centers[:, 0],
    centers[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("K-Means Customer Segmentation")
plt.xlabel("Annual Income (Scaled)")
plt.ylabel("Spending Score (Scaled)")
plt.legend()
plt.show()

print("\nFinal Dataset with Clusters")
print(df.head(10))

df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("\nClustered dataset saved successfully.")


# In[ ]:




