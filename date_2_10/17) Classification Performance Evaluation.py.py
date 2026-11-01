#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, ConfusionMatrixDisplay

df = pd.read_csv("Breast_Cancer.csv")

df.columns = df.columns.str.strip().str.lower()
df = df.dropna(axis=1, how="all")
df = df.loc[:, ~df.columns.str.contains("^unnamed")]
df = df.drop_duplicates()

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

X = X.select_dtypes(include="number")
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median())

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("Classification Performance Evaluation")
print("--------------------------------------")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-Score:", round(f1, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=encoder.classes_,
    cmap="Blues"
)

plt.title("Logistic Regression Confusion Matrix")
plt.show()

metrics = pd.Series({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1
})

plt.figure(figsize=(8, 5))
metrics.plot(kind="bar")
plt.title("Classification Performance Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.show()


# In[ ]:




