"""
===============================================================================
ML1 : Predict the price of the Uber ride from a given pickup point to the
      agreed drop-off location. Perform following tasks:
        1. Pre-process the dataset.
        2. Identify outliers.
        3. Check the correlation.
        4. Implement linear regression and random forest regression models.
        5. Evaluate the models and compare their respective scores like R2,
           RMSE, etc.
      Dataset: https://www.kaggle.com/datasets/yasserh/uber-fares-dataset
      (download "uber.csv" and keep it in the same folder as this file)
===============================================================================

THEORY
------
1. MACHINE LEARNING (Unit I)
   "A computer program is said to learn from experience E with respect to
   some task T and performance measure P, if its performance at T, as measured
   by P, improves with experience E." - Tom Mitchell.
   Traditional programming: Data + Program -> Output
   Machine learning       : Data + Output  -> Program (model)
   AI (broad goal of intelligent machines) > ML (learning from data) >
   Deep Learning (multi-layer neural nets). Data Science = extracting insight
   from data using statistics + ML + domain knowledge.
   Types of learning: SUPERVISED (labelled data: regression, classification),
   UNSUPERVISED (unlabelled: clustering), SEMI-SUPERVISED, REINFORCEMENT
   (agent learns via rewards).
   This problem is SUPERVISED REGRESSION (target fare_amount is continuous).

2. DATA PRE-PROCESSING (Unit II - Feature Engineering)
   - Handling missing values: drop rows, or impute with mean/median/mode.
   - Removing invalid data: fare <= 0, coordinates outside valid ranges,
     passenger_count = 0 or > 6.
   - Feature extraction: from pickup_datetime derive hour, day, month, year,
     weekday; from coordinates derive trip DISTANCE using the HAVERSINE
     formula:
        a = sin^2(dlat/2) + cos(lat1) cos(lat2) sin^2(dlon/2)
        d = 2 R asin(sqrt(a)),   R = 6371 km
   - Scaling: Standardization z = (x - mean)/std, Normalization (min-max)
     x' = (x - min)/(max - min). Needed for distance/gradient based models
     (linear regression with GD, KNN, SVM, NN), not for tree models.

3. OUTLIERS (Unit V - Outlier analysis)
   An outlier is an observation far away from the rest of the data.
   Detection: box-plot / IQR rule
        IQR = Q3 - Q1 ; outlier if x < Q1 - 1.5 IQR or x > Q3 + 1.5 IQR
   or Z-score |z| > 3, or Isolation Forest / Local Outlier Factor.
   Treatment: remove, or CAP (clip) values to the IQR limits (done here).

4. CORRELATION
   Pearson correlation r = cov(X,Y) / (sigma_X sigma_Y), -1 <= r <= 1.
   r ~ +1 strong positive, r ~ -1 strong negative, r ~ 0 no linear relation.
   Visualised using a heat-map; helps feature selection and detecting
   multicollinearity.

5. LINEAR REGRESSION (Unit III)
   Models y = b0 + b1 x1 + ... + bn xn + e.
   Parameters chosen to minimise Mean Squared Error (Ordinary Least Squares):
        J(b) = (1/m) SUM (y_i - y_hat_i)^2
   Closed-form: b = (X^T X)^-1 X^T y, or iteratively by GRADIENT DESCENT.
   Regularised variants: RIDGE (L2 penalty lambda SUM b^2, shrinks weights),
   LASSO (L1 penalty lambda SUM |b|, can make weights exactly 0 -> feature
   selection).

6. RANDOM FOREST REGRESSION (Unit IV - Ensemble learning)
   ENSEMBLE: combine many weak learners to get a strong learner.
   BAGGING (Bootstrap AGGregatING): train each model on a bootstrap sample
   (sampling with replacement) and average predictions -> reduces variance.
   RANDOM FOREST = bagging of decision trees + random subset of features at
   each split. Prediction = average of all trees (regression).
   Handles non-linearity, robust to outliers, no scaling needed.

7. BIAS, VARIANCE, OVER/UNDER-FITTING (Unit III)
   Bias = error from wrong assumptions (underfitting: model too simple).
   Variance = error from sensitivity to training data (overfitting: model
   too complex, memorises noise). Goal: good GENERALIZATION (low test error);
   Total error = Bias^2 + Variance + irreducible noise.

8. EVALUATION METRICS FOR REGRESSION
   MAE  = (1/n) SUM |y - y_hat|
   MSE  = (1/n) SUM (y - y_hat)^2
   RMSE = sqrt(MSE)   (same units as target, penalises large errors)
   R^2  = 1 - SS_res/SS_tot   (fraction of variance explained, 1 = perfect)

CONCLUSION: After cleaning, outlier capping and distance feature extraction,
Random Forest generally gives higher R2 and lower RMSE than Linear
Regression because fare depends non-linearly on distance/time.
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "uber.csv")

# %% 1. Read dataset ---------------------------------------------------------
df = pd.read_csv(DATA)
print("Shape:", df.shape)
print(df.head())
df.info()

# %% 2. Pre-processing -------------------------------------------------------
df = df.drop(columns=[c for c in ["Unnamed: 0", "key"] if c in df.columns])
print("\nMissing values:\n", df.isnull().sum())
df = df.dropna()

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna(subset=["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day"] = df["pickup_datetime"].dt.day
df["month"] = df["pickup_datetime"].dt.month
df["year"] = df["pickup_datetime"].dt.year
df["weekday"] = df["pickup_datetime"].dt.dayofweek
df = df.drop(columns=["pickup_datetime"])

# remove impossible values (rides are in New York City: lat ~40-42,
# lon ~-75 to -72; some rows have swapped / zero coordinates)
df = df[(df["fare_amount"] > 0) &
        (df["passenger_count"] > 0) & (df["passenger_count"] <= 6) &
        df["pickup_latitude"].between(40, 42) &
        df["dropoff_latitude"].between(40, 42) &
        df["pickup_longitude"].between(-75, -72) &
        df["dropoff_longitude"].between(-75, -72)]


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    a = (np.sin((lat2 - lat1) / 2) ** 2 +
         np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 2 * 6371 * np.arcsin(np.sqrt(a))


df["distance_km"] = haversine(df["pickup_longitude"], df["pickup_latitude"],
                              df["dropoff_longitude"], df["dropoff_latitude"])
df = df[df["distance_km"] > 0]
print("\nShape after cleaning:", df.shape)
print(df.describe().T)

# %% 3. Identify outliers (box plots + IQR) ----------------------------------
num_cols = ["fare_amount", "distance_km", "pickup_longitude",
            "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
            "passenger_count"]
df[num_cols].plot(kind="box", subplots=True, layout=(2, 4), figsize=(16, 7),
                  title="Box plots BEFORE outlier treatment")
plt.tight_layout()
plt.show()


def iqr_bounds(s):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


print("\nOutlier count per column (IQR rule):")
for c in num_cols:
    lo, hi = iqr_bounds(df[c])
    print(f"  {c:<20} {((df[c] < lo) | (df[c] > hi)).sum()}")
    if c != "passenger_count":          # discrete 1-6, already validated
        df[c] = df[c].clip(lo, hi)      # cap outliers

df[num_cols].plot(kind="box", subplots=True, layout=(2, 4), figsize=(16, 7),
                  title="Box plots AFTER outlier treatment")
plt.tight_layout()
plt.show()

# %% 4. Correlation ----------------------------------------------------------
corr = df.corr()
print("\nCorrelation with fare_amount:\n",
      corr["fare_amount"].sort_values(ascending=False))
plt.figure(figsize=(11, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation heat-map")
plt.show()

# %% 5. Train-test split & scaling -------------------------------------------
X = df.drop(columns=["fare_amount"])
y = df["fare_amount"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# %% 6. Models ---------------------------------------------------------------
lr = LinearRegression()
lr.fit(X_train_s, y_train)
y_pred_lr = lr.predict(X_test_s)

rf = RandomForestRegressor(n_estimators=100, max_depth=15, n_jobs=-1,
                           random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


# %% 7. Evaluation -----------------------------------------------------------
def evaluate(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"Model": name,
            "R2": r2_score(y_true, y_pred),
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mse,
            "RMSE": np.sqrt(mse)}


results = pd.DataFrame([evaluate("Linear Regression", y_test, y_pred_lr),
                        evaluate("Random Forest", y_test, y_pred_rf)])
print("\nModel comparison:\n", results.to_string(index=False))

print("\nLinear regression coefficients:")
for f, c in zip(X.columns, lr.coef_):
    print(f"  {f:<20} {c:8.3f}")

print("\nRandom forest feature importance:")
for f, imp in sorted(zip(X.columns, rf.feature_importances_),
                     key=lambda t: -t[1]):
    print(f"  {f:<20} {imp:.3f}")

plt.figure(figsize=(12, 5))
for i, (name, pred) in enumerate([("Linear Regression", y_pred_lr),
                                  ("Random Forest", y_pred_rf)]):
    plt.subplot(1, 2, i + 1)
    plt.scatter(y_test, pred, s=3, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r")
    plt.xlabel("Actual fare")
    plt.ylabel("Predicted fare")
    plt.title(name)
plt.tight_layout()
plt.show()

best = results.loc[results["R2"].idxmax(), "Model"]
print(f"\nBetter model: {best}")
