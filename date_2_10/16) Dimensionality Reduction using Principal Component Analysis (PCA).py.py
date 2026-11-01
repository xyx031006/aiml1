#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

df = pd.read_csv("Breast_Cancer.csv")

df.columns = df.columns.str.strip().str.lower()

df = df.dropna(axis=1, how="all")

df = df.loc[:, ~df.columns.str.contains("^unnamed")]

target_column = "diagnosis"

X = df.drop(columns=[target_column])
y = df[target_column]

X = X.select_dtypes(include="number")

X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["Principal Component 1", "Principal Component 2"]
)

pca_df["Diagnosis"] = y.values

print("Original Dataset Shape:", df.shape)
print("Original Number of Features:", X.shape[1])
print("Reduced Number of Features:", X_pca.shape[1])

print("\nReduced Dataset:")
print(pca_df.head())

print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print(
    "\nTotal Variance Retained:",
    round(sum(pca.explained_variance_ratio_) * 100, 2), "%"
)

plt.figure(figsize=(9, 6))

for label in encoder.classes_:
    subset = pca_df[pca_df["Diagnosis"] == label]
    plt.scatter(
        subset["Principal Component 1"],
        subset["Principal Component 2"],
        label=label,
        alpha=0.7
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA - Breast Cancer Dataset")
plt.legend()
plt.grid(True)
plt.show()

pca_df.to_csv("Breast_Cancer_PCA.csv", index=False)

print("\nPCA reduced dataset saved successfully.")


# In[ ]:




