"""
===============================================================================
ML3 : Given a bank customer, build a neural network-based classifier that can
      determine whether they will leave or not in the next 6 months.
      Dataset: https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling
      (download "Churn_Modelling.csv" and keep it in the same folder)
      Steps:
        1. Read the dataset.
        2. Distinguish the feature and target set and divide the data set into
           training and test sets.
        3. Normalize the train and test data.
        4. Initialize and build the model. Identify the points of improvement
           and implement the same.
        5. Print the accuracy score and confusion matrix.
      Uses TensorFlow/Keras if installed (pip install tensorflow), otherwise
      falls back to scikit-learn's MLPClassifier (same architecture).
===============================================================================

THEORY
------
1. ARTIFICIAL NEURAL NETWORK (Unit VI)
   Inspired by biological neurons. An artificial neuron computes
        z = SUM w_i x_i + b ,   output a = f(z)
   where w = weights, b = bias, f = activation function.
   - SINGLE LAYER PERCEPTRON: input layer directly connected to output;
     can only learn LINEARLY SEPARABLE functions (cannot learn XOR).
     Update rule: w = w + eta (target - output) x
   - MULTILAYER PERCEPTRON (MLP): input layer + one or more HIDDEN layers +
     output layer; fully connected (dense); can approximate any continuous
     function (universal approximation theorem).

2. ACTIVATION FUNCTIONS
   Sigmoid  f(z) = 1/(1 + e^-z), range (0,1) - used for binary output.
   Tanh     f(z) = (e^z - e^-z)/(e^z + e^-z), range (-1,1).
   ReLU     f(z) = max(0, z) - fast, avoids vanishing gradient; used in
            hidden layers.
   Softmax  f(z_i) = e^z_i / SUM e^z_j - multiclass probabilities.
   Step / linear functions are used in simple perceptrons / regression.

3. BACK-PROPAGATION LEARNING
   a) FORWARD PASS: compute outputs layer by layer.
   b) LOSS: binary cross-entropy  L = -[y log(p) + (1-y) log(1-p)].
   c) BACKWARD PASS: compute gradient dL/dw of every weight using the CHAIN
      RULE, propagating error from output towards input.
   d) UPDATE: w = w - eta * dL/dw (gradient descent; optimisers SGD, Adam).
   One pass over the whole training set = 1 EPOCH; weights are updated after
   every BATCH.

4. OTHER NETWORKS (syllabus)
   - Functional Link ANN: expands inputs with non-linear functions
     (x, x^2, sin x ...) so a single layer can learn non-linear mappings.
   - Radial Basis Function Network: hidden units use Gaussian functions of
     distance from centres; output is linear combination.
   - RNN: has feedback loops / memory -> sequence data (text, time series).
   - CNN: convolution + pooling layers -> images.

5. PRE-PROCESSING FOR THIS DATASET
   - Drop identifiers RowNumber, CustomerId, Surname (no predictive value).
   - Encode categorical data: Gender -> Label encoding (0/1); Geography ->
     One-hot encoding (France/Germany/Spain).
   - Normalize/standardize features (StandardScaler) so that gradient
     descent converges faster and no feature dominates.
   - Target: Exited (1 = customer left, 0 = stayed). Dataset is IMBALANCED
     (~20% churn), so recall of class 1 is important.

6. POINTS OF IMPROVEMENT (implemented in the improved model)
   - More neurons / extra hidden layer (more capacity).
   - Dropout (Keras) / L2 regularisation alpha (sklearn) -> reduce overfitting.
   - Adam optimiser with tuned learning rate.
   - Early stopping on validation loss.
   - Class weights / threshold tuning for imbalanced data.
   - Hyper-parameter tuning: epochs, batch size.

7. METRICS: Accuracy = (TP+TN)/total ; Confusion matrix [[TN FP][FN TP]];
   Precision, Recall, F1 (see ML2).

CONCLUSION: An MLP with ReLU hidden layers and sigmoid output predicts churn
with ~85-86% accuracy; regularisation and early stopping improve
generalization.
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
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Churn_Modelling.csv")

try:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras
    USE_KERAS = True
except ImportError:
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    warnings.filterwarnings("ignore", category=ConvergenceWarning)  # base model is intentionally short
    USE_KERAS = False
print("Backend:", "TensorFlow/Keras" if USE_KERAS else "scikit-learn MLP")

# %% 1. Read the dataset ------------------------------------------------------
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print(df.head())
print("Missing values:", df.isnull().sum().sum())

# %% 2. Features / target, encoding, split ------------------------------------
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
df = pd.get_dummies(df, columns=["Geography"], drop_first=True, dtype=int)

X = df.drop(columns=["Exited"])
y = df["Exited"]
print("\nFeatures:", list(X.columns))
print("Target distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# %% 3. Normalize ------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# %% 4. Build models ----------------------------------------------------------
def build_keras(improved):
    model = keras.Sequential([keras.Input(shape=(X_train.shape[1],))])
    if improved:
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Dense(16, activation="relu"))
        model.add(keras.layers.Dropout(0.2))
    else:
        model.add(keras.layers.Dense(6, activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    opt = keras.optimizers.Adam(learning_rate=0.001 if improved else 0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def train_and_predict(improved):
    if USE_KERAS:
        model = build_keras(improved)
        model.summary()
        cb = ([keras.callbacks.EarlyStopping(patience=10,
                                             restore_best_weights=True)]
              if improved else [])
        hist = model.fit(X_train, y_train, validation_split=0.2,
                         epochs=100 if improved else 20, batch_size=32,
                         callbacks=cb, verbose=0)
        prob = model.predict(X_test, verbose=0).ravel()
        return (prob > 0.5).astype(int), hist.history["loss"], \
            hist.history["val_loss"]
    model = MLPClassifier(
        hidden_layer_sizes=(32, 16) if improved else (6,),
        activation="relu", solver="adam",
        alpha=1e-3 if improved else 1e-4,              # L2 regularisation
        learning_rate_init=0.001 if improved else 0.01,
        max_iter=300 if improved else 20,
        early_stopping=improved, validation_fraction=0.2,
        n_iter_no_change=10, random_state=42)
    model.fit(X_train, y_train)
    return model.predict(X_test), model.loss_curve_, \
        getattr(model, "validation_scores_", None)


results = {}
for label, improved in [("Base model (1 hidden layer, 6 units)", False),
                        ("Improved model (32-16, regularised, early stop)",
                         True)]:
    print(f"\n===== {label} =====")
    y_pred, loss, val = train_and_predict(improved)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    results[label] = (acc, cm)

    # %% 5. Accuracy & confusion matrix
    print(f"Accuracy score: {acc:.4f}")
    print("Confusion matrix:\n", cm)
    print(classification_report(y_test, y_pred,
                                target_names=["Stayed", "Exited"],
                                zero_division=0))
    plt.plot(loss, label=f"train loss - {'improved' if improved else 'base'}")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training loss")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (label, (acc, cm)) in zip(axes, results.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stayed", "Exited"],
                yticklabels=["Stayed", "Exited"])
    ax.set_title(f"{label.split('(')[0]} acc={acc:.3f}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.show()
