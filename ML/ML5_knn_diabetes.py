"""
===============================================================================
ML5 : Implement K-Nearest Neighbors algorithm on diabetes.csv dataset.
      Compute confusion matrix, accuracy, error rate, precision and recall on
      the given dataset.
      Dataset: https://www.kaggle.com/datasets/abdallamahgoub/diabetes
      (download "diabetes.csv" and keep it in the same folder as this file)
===============================================================================

THEORY
------
1. DATASET (Pima Indians Diabetes)
   768 records, 8 features: Pregnancies, Glucose, BloodPressure,
   SkinThickness, Insulin, BMI, Pedigree (DiabetesPedigreeFunction), Age;
   target Outcome (1 = diabetic, 0 = non-diabetic).
   Several columns contain 0 where 0 is physically impossible (Glucose,
   BloodPressure, SkinThickness, Insulin, BMI) -> treated as MISSING VALUES and
   replaced by the column MEDIAN (imputation, Unit II).

2. K-NEAREST NEIGHBOURS ALGORITHM
   Non-parametric, lazy (instance-based) supervised learner.
       1. Choose k.
       2. For a test point compute distance to every training point
          (Euclidean d = sqrt(SUM (x_i - y_i)^2)).
       3. Select the k closest points.
       4. Assign the class that is in MAJORITY among them
          (regression: average of their values).
   Properties: no training phase; prediction O(n.d); sensitive to scale
   (so features are STANDARDIZED) and to irrelevant features; "curse of
   dimensionality". k chosen by validation, commonly sqrt(n), odd for binary.
   Small k -> overfitting (high variance), large k -> underfitting (high bias).
   Weighted KNN gives closer neighbours more weight (1/d).

3. CONFUSION MATRIX & METRICS
                         Predicted 0   Predicted 1
        Actual 0            TN            FP
        Actual 1            FN            TP
   Accuracy   = (TP + TN) / (TP + TN + FP + FN)
   Error rate = (FP + FN) / total = 1 - Accuracy
   Precision  = TP / (TP + FP)   -> exactness of positive predictions
   Recall     = TP / (TP + FN)   -> sensitivity / true positive rate
   F1         = 2 PR / (P + R)
   Specificity = TN / (TN + FP)
   In medical diagnosis RECALL is critical: a False Negative means a diabetic
   patient is missed.

CONCLUSION: KNN with scaled features and median-imputed zeros gives about
70-78% accuracy on the diabetes dataset; metrics are computed manually from the
confusion matrix and verified with scikit-learn.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             precision_score, recall_score,
                             classification_report)

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")

# %% 1. Read & pre-process ----------------------------------------------------
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print(df.head())
print(df.describe().T)

target = "Outcome"
zero_invalid = [c for c in ["Glucose", "BloodPressure", "SkinThickness",
                            "Insulin", "BMI"] if c in df.columns]
print("\nZero counts (invalid values):\n", (df[zero_invalid] == 0).sum())
for c in zero_invalid:
    df[c] = df[c].replace(0, np.nan)
    df[c] = df[c].fillna(df[c].median())

X = df.drop(columns=[target])
y = df[target]
print("\nClass distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% 2. Choose k --------------------------------------------------------------
errors = []
ks = range(1, 31, 2)                    # odd k avoids ties in voting
for k in ks:
    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    errors.append(1 - accuracy_score(y_test, m.predict(X_test)))
best_k = list(ks)[int(np.argmin(errors))]
print("\nBest k (lowest error) =", best_k)

plt.plot(list(ks), errors, marker="o")
plt.xlabel("k")
plt.ylabel("Error rate")
plt.title("Error rate vs k")
plt.show()

# %% 3. Final KNN model -------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# %% 4. Metrics ---------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
total = tn + fp + fn + tp

print("\nConfusion matrix:\n", cm)
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
print(f"Accuracy   = (TP+TN)/total = {(tp + tn) / total:.4f}")
print(f"Error rate = (FP+FN)/total = {(fp + fn) / total:.4f}")
print(f"Precision  = TP/(TP+FP)    = {tp / (tp + fp):.4f}")
print(f"Recall     = TP/(TP+FN)    = {tp / (tp + fn):.4f}")

print("\nVerification with sklearn:")
print(f"Accuracy  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision {precision_score(y_test, y_pred):.4f}")
print(f"Recall    {recall_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred,
                            target_names=["Non-diabetic", "Diabetic"]))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-diabetic", "Diabetic"],
            yticklabels=["Non-diabetic", "Diabetic"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"KNN (k={best_k}) confusion matrix")
plt.show()
