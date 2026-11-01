#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Titanic.csv")

df.columns = df.columns.str.strip().str.lower()

print("Original Dataset")
print(df.head())

print("\nDataset Shape:", df.shape)

df = df.drop_duplicates()

df = df.drop(
    columns=["passengerid", "name", "ticket", "cabin"],
    errors="ignore"
)

target = "survived"

if target not in df.columns:
    raise ValueError("Survived column not found in Titanic dataset.")

df = df.dropna(subset=[target])

print("\nMissing Values Before Handling:")
print(df.isnull().sum())

X = df.drop(columns=[target])
y = df[target]

numeric_columns = X.select_dtypes(include=np.number).columns
categorical_columns = X.select_dtypes(exclude=np.number).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for col in numeric_columns:
    median_value = X_train[col].median()
    X_train[col] = X_train[col].fillna(median_value)
    X_test[col] = X_test[col].fillna(median_value)

for col in categorical_columns:
    mode_value = X_train[col].mode()[0]
    X_train[col] = X_train[col].fillna(mode_value)
    X_test[col] = X_test[col].fillna(mode_value)

print("\nMissing Values After Handling:")
print(X_train.isnull().sum())

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

X_train, X_test = X_train.align(
    X_test, join="left", axis=1, fill_value=0
)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nStatistical Analysis")
print(df.describe(include="all"))

print("\nMean:")
print(df.select_dtypes(include=np.number).mean())

print("\nMedian:")
print(df.select_dtypes(include=np.number).median())

print("\nStandard Deviation:")
print(df.select_dtypes(include=np.number).std())

plt.figure(figsize=(10, 6))
df.select_dtypes(include=np.number).hist(figsize=(12, 8), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.tight_layout()
plt.show()

numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(12, 6))
sns.boxplot(data=numeric_data)
plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nTraining and Testing Data")
print("Training Features Shape:", X_train_scaled.shape)
print("Testing Features Shape:", X_test_scaled.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

print("\nData Preprocessing Completed Successfully!")

processed_train = pd.DataFrame(
    X_train_scaled, columns=X_train.columns
)
processed_train[target] = y_train.reset_index(drop=True)

processed_test = pd.DataFrame(
    X_test_scaled, columns=X_test.columns
)
processed_test[target] = y_test.reset_index(drop=True)

processed_train.to_csv("Titanic_Train_Preprocessed.csv", index=False)
processed_test.to_csv("Titanic_Test_Preprocessed.csv", index=False)

print("\nPreprocessed training and testing datasets saved successfully.")


# In[ ]:




