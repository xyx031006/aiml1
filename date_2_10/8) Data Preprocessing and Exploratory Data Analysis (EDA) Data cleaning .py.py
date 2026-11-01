#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Titanic.csv")

print("========== TITANIC DATASET ==========")

print("\nFirst 5 Records:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\n========== DATA CLEANING ==========")

print("\nDuplicate Records:", df.duplicated().sum())

df = df.drop_duplicates()

print("Shape After Removing Duplicates:", df.shape)

print("\n========== MISSING VALUE HANDLING ==========")

print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

for col in df.columns:

    if df[col].isnull().sum() > 0:

        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])

        else:
            df[col] = df[col].fillna(df[col].median())

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

print("\n========== FEATURE SCALING ==========")

numeric_cols = df.select_dtypes(include=np.number).columns

scaled_df = df.copy()

for col in numeric_cols:

    minimum = df[col].min()
    maximum = df[col].max()

    if maximum != minimum:
        scaled_df[col] = (df[col] - minimum) / (maximum - minimum)

print("\nScaled Numerical Features:")
print(scaled_df[numeric_cols].head())

print("\n========== ENCODING TECHNIQUES ==========")

encoded_df = pd.get_dummies(
    df,
    columns=df.select_dtypes(include="object").columns,
    dtype=int
)

print("\nEncoded Dataset:")
print(encoded_df.head())

print("\n========== STATISTICAL ANALYSIS ==========")

print("\nDescriptive Statistics:")
print(df.describe(include="all"))

print("\nMean:")
print(df.select_dtypes(include=np.number).mean())

print("\nMedian:")
print(df.select_dtypes(include=np.number).median())

print("\nMode:")
print(df.mode().iloc[0])

print("\nStandard Deviation:")
print(df.select_dtypes(include=np.number).std())

print("\n========== EXPLORATORY DATA ANALYSIS ==========")

if "Survived" in df.columns:

    print("\nSurvival Count:")
    print(df["Survived"].value_counts())

    df["Survived"].value_counts().plot(
        kind="bar",
        title="Titanic Survival Count"
    )

    plt.xlabel("Survived (0 = No, 1 = Yes)")
    plt.ylabel("Number of Passengers")
    plt.show()

if "Pclass" in df.columns:

    df["Pclass"].value_counts().sort_index().plot(
        kind="bar",
        title="Passenger Class Distribution"
    )

    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.show()

if "Age" in df.columns:

    df["Age"].plot(
        kind="hist",
        bins=20,
        title="Age Distribution"
    )

    plt.xlabel("Age")
    plt.show()

print("\n========== CORRELATION ANALYSIS ==========")

correlation = df.select_dtypes(include=np.number).corr()

print(correlation)

plt.figure(figsize=(10, 6))
plt.imshow(correlation, cmap="coolwarm", aspect="auto")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Titanic Dataset Correlation Matrix")
plt.tight_layout()
plt.show()

print("\n========== FINAL RESULT ==========")

print("Data Cleaning Completed")
print("Missing Values Handled")
print("Feature Scaling Completed")
print("Categorical Encoding Completed")
print("Statistical Analysis Completed")
print("EDA Completed Successfully!")


# In[ ]:




