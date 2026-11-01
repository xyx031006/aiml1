#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

df = pd.read_csv("Mall_Customers.csv")

df.columns = df.columns.str.strip()

print("Dataset Information")
print(df.head())
print("\nDataset Shape:", df.shape)

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

linked = linkage(X_scaled, method="ward")

plt.figure(figsize=(12, 6))
dendrogram(linked, truncate_mode="lastp", p=30)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customer Groups")
plt.ylabel("Euclidean Distance")
plt.show()

optimal_k = 5

model = AgglomerativeClustering(
    n_clusters=optimal_k,
    linkage="ward"
)

df["Cluster"] = model.fit_predict(X_scaled)

print("\nCustomer Cluster Analysis")
print(
    df.groupby("Cluster")[
        ["Annual Income (k$)", "Spending Score (1-100)"]
    ].mean().round(2)
)

print("\nNumber of Customers in Each Cluster")
print(df["Cluster"].value_counts().sort_index())

score = silhouette_score(X_scaled, df["Cluster"])
print("\nSilhouette Score:", round(score, 4))

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Hierarchical Clustering - Customer Segmentation")
plt.show()

print("\nFinal Dataset with Clusters")
print(df.head(10))

df.to_csv("Mall_Customers_Hierarchical_Clusters.csv", index=False)

print("\nClustered dataset saved successfully.")


# In[ ]:




