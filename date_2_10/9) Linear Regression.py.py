#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("California_Housing.csv")

print("========== CALIFORNIA HOUSING DATASET ==========")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\n========== DATA PREPROCESSING ==========")

target = "median_house_value"

if target not in df.columns:
    raise ValueError("Target column not found. Available columns: " + str(list(df.columns)))

df = df.drop_duplicates()
df = df.dropna(subset=[target])

X = df.drop(columns=[target])
y = df[target]

X = pd.get_dummies(X, drop_first=True, dtype=int)

X = X.fillna(X.median())

print("Features:", X.shape)
print("Target:", y.shape)

print("\n========== TRAIN TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

print("\n========== LINEAR REGRESSION MODEL ==========")

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Model Training Completed Successfully!")

print("\n========== MODEL EVALUATION ==========")

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("R2 Score:", r2)

print("\n========== ACTUAL VS PREDICTED VALUES ==========")

comparison = pd.DataFrame({
    "Actual Values": y_test.values[:10],
    "Predicted Values": y_pred[:10]
})

print(comparison)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual House Values")
plt.ylabel("Predicted House Values")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

print("\nLinear Regression Completed Successfully!")


# In[ ]:




