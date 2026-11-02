"""
===============================================================================
ML2 : Classify the email using the binary classification method. Email Spam
      detection has two states: a) Normal State - Not Spam, b) Abnormal State
      - Spam. Use K-Nearest Neighbors and Support Vector Machine for
      classification. Analyze their performance.
      Dataset: https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
      (download "emails.csv" and keep it in the same folder as this file)
===============================================================================

THEORY
------
1. CLASSIFICATION (Unit IV)
   Supervised learning where the target is a discrete class label.
   BINARY classification: 2 classes (Spam = 1 / Not spam = 0).
   MULTICLASS: > 2 classes; handled by
     - One-vs-All (OvR): one classifier per class vs rest (k classifiers)
     - One-vs-One (OvO): one classifier per pair of classes (k(k-1)/2)
   BALANCED vs IMBALANCED: if one class dominates, accuracy is misleading;
   use precision/recall/F1, stratified splits, resampling, class weights.

2. DATASET
   5172 emails; each row = an email, 3000 columns = frequency of the 3000
   most common words (BAG-OF-WORDS representation), last column Prediction
   (1 = spam, 0 = not spam). "Email No." is an ID and is dropped.

3. K-NEAREST NEIGHBOURS (KNN)
   A LAZY, NON-PARAMETRIC, instance-based learner - no explicit training;
   stores the data and classifies a new point by MAJORITY VOTE of its k
   nearest training points.
   Distance metrics: Euclidean sqrt(SUM (xi - yi)^2), Manhattan SUM |xi - yi|,
   Minkowski (SUM |xi - yi|^p)^(1/p).
   Choice of k: small k -> noisy, overfits (high variance); large k ->
   smooth, underfits (high bias). Odd k avoids ties in binary problems.
   Feature scaling is essential because distance depends on magnitude.
   Prediction cost O(n d) per query (n samples, d features).

4. SUPPORT VECTOR MACHINE (SVM)
   Finds the HYPERPLANE w.x + b = 0 that separates the classes with the
   MAXIMUM MARGIN. Margin = 2/||w||. The training points lying on the margin
   boundaries are the SUPPORT VECTORS.
   Hard margin: minimise ||w||^2/2 s.t. y_i (w.x_i + b) >= 1.
   Soft margin: allows violations with slack xi_i, penalty C:
        minimise ||w||^2/2 + C SUM xi_i
   (large C -> narrow margin, fewer errors on train; small C -> wider margin)
   KERNEL TRICK: maps data to higher dimension implicitly for non-linear
   boundaries. Kernels: linear, polynomial (x.y + c)^d, RBF
   exp(-gamma ||x - y||^2), sigmoid.
   Works well with high-dimensional sparse data such as text.

5. EVALUATION METRICS
   Confusion matrix:            Predicted
                              Neg     Pos
               Actual Neg     TN      FP
               Actual Pos     FN      TP
   Accuracy  = (TP + TN) / (TP + TN + FP + FN)
   Precision = TP / (TP + FP)   (of predicted spam, how many are spam)
   Recall    = TP / (TP + FN)   (of real spam, how many were caught)
   F1-score  = 2 P R / (P + R)  (harmonic mean)
   Micro-average: pool TP/FP/FN of all classes then compute.
   Macro-average: compute per class then take the unweighted mean.
   CROSS-VALIDATION (k-fold): split data into k folds, train on k-1 and test
   on the remaining one, repeat k times, average -> reliable estimate.

CONCLUSION: Both classifiers detect spam well; SVM (linear / RBF) usually
achieves higher accuracy and precision than KNN on high-dimensional word-count
features, while KNN is simple but slower at prediction time.
===============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "emails.csv")

# %% 1. Read & pre-process ----------------------------------------------------
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print(df.iloc[:5, :8])
print("Missing values:", df.isnull().sum().sum())
df = df.dropna()

X = df.drop(columns=["Email No.", "Prediction"], errors="ignore")
y = df["Prediction"]
print("\nClass distribution (0 = Not spam, 1 = Spam):\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# %% 2. KNN - choose k -------------------------------------------------------
print("\nKNN accuracy for different k:")
k_values = range(1, 16, 2)
k_acc = []
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, knn_k.predict(X_test_s))
    k_acc.append(acc)
    print(f"  k = {k:2d}  accuracy = {acc:.4f}")
best_k = list(k_values)[int(np.argmax(k_acc))]
print("Best k =", best_k)

plt.plot(list(k_values), k_acc, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("KNN: accuracy vs k")
plt.show()

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_s, y_train)
y_pred_knn = knn.predict(X_test_s)

# %% 3. SVM -------------------------------------------------------------------
svm_lin = SVC(kernel="linear", C=1.0)
svm_lin.fit(X_train_s, y_train)
y_pred_lin = svm_lin.predict(X_test_s)

svm_rbf = SVC(kernel="rbf", C=10, gamma="scale")
svm_rbf.fit(X_train_s, y_train)
y_pred_rbf = svm_rbf.predict(X_test_s)


# %% 4. Performance analysis -------------------------------------------------
def metrics(name, y_true, y_pred):
    return {"Model": name,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred),
            "Recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)}


preds = {f"KNN (k={best_k})": y_pred_knn,
         "SVM (linear)": y_pred_lin,
         "SVM (RBF)": y_pred_rbf}

results = pd.DataFrame([metrics(n, y_test, p) for n, p in preds.items()])
print("\nPerformance comparison:\n", results.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, p) in zip(axes, preds.items()):
    print(f"\n{name}\nConfusion matrix:\n{confusion_matrix(y_test, p)}")
    print(classification_report(y_test, p,
                                target_names=["Not Spam", "Spam"]))
    sns.heatmap(confusion_matrix(y_test, p), annot=True, fmt="d",
                cmap="Blues", ax=ax,
                xticklabels=["Not Spam", "Spam"],
                yticklabels=["Not Spam", "Spam"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()

# 5-fold cross-validation for a more reliable comparison
print("5-fold cross-validation accuracy:")
for name, model in [(f"KNN (k={best_k})", KNeighborsClassifier(best_k)),
                    ("SVM (linear)", SVC(kernel="linear"))]:
    scores = cross_val_score(model, scaler.transform(X), y, cv=5)
    print(f"  {name:<14} mean = {scores.mean():.4f}  std = {scores.std():.4f}")

print("\nBest model by F1:", results.loc[results["F1"].idxmax(), "Model"])
