#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

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
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "SVM": SVC(kernel="linear", C=1),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=50,
        random_state=42
    )
}

results = []

for name, model in models.items():

    if name == "SVM":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append([name, accuracy, precision, recall, f1])

    print("\n", name, "Performance")
    print("--------------------------------")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-Score:", round(f1, 4))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, target_names=encoder.classes_
    ))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=encoder.classes_
    )
    disp.plot(cmap="Blues")
    plt.title(name + " Confusion Matrix")
    plt.show()

comparison = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nFinal Model Comparison")
print(comparison.to_string(index=False))

best_model = comparison.loc[comparison["F1-Score"].idxmax()]

print("\nBest Performing Model:", best_model["Model"])
print("Best F1-Score:", round(best_model["F1-Score"], 4))

comparison.set_index("Model").plot(
    kind="bar", figsize=(10, 6)
)

plt.title("SVM vs Random Forest vs AdaBoost")
plt.ylabel("Performance Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.show()


# In[ ]:




