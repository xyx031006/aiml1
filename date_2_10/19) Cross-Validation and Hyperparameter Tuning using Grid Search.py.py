#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

default_model = pipeline.fit(X_train, y_train)
before_pred = default_model.predict(X_test)

before_accuracy = accuracy_score(y_test, before_pred)
before_precision = precision_score(y_test, before_pred, average="weighted")
before_recall = recall_score(y_test, before_pred, average="weighted")
before_f1 = f1_score(y_test, before_pred, average="weighted")

print("Performance Before Optimization")
print("--------------------------------")
print("Accuracy:", round(before_accuracy, 4))
print("Precision:", round(before_precision, 4))
print("Recall:", round(before_recall, 4))
print("F1-Score:", round(before_f1, 4))

param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__kernel": ["linear", "rbf"],
    "svm__gamma": ["scale", "auto"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
after_pred = best_model.predict(X_test)

after_accuracy = accuracy_score(y_test, after_pred)
after_precision = precision_score(y_test, after_pred, average="weighted")
after_recall = recall_score(y_test, after_pred, average="weighted")
after_f1 = f1_score(y_test, after_pred, average="weighted")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation Accuracy:")
print(round(grid_search.best_score_, 4))

print("\nPerformance After Optimization")
print("--------------------------------")
print("Accuracy:", round(after_accuracy, 4))
print("Precision:", round(after_precision, 4))
print("Recall:", round(after_recall, 4))
print("F1-Score:", round(after_f1, 4))

comparison = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Before Optimization": [
        before_accuracy, before_precision, before_recall, before_f1
    ],
    "After Optimization": [
        after_accuracy, after_precision, after_recall, after_f1
    ]
})

print("\nBefore and After Optimization Comparison")
print(comparison.to_string(index=False))

print("\nClassification Report After Optimization:")
print(classification_report(y_test, after_pred))

comparison.set_index("Metric").plot(
    kind="bar", figsize=(9, 5)
)

plt.title("SVM Performance Before and After Grid Search")
plt.ylabel("Performance Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.show()


# In[ ]:




