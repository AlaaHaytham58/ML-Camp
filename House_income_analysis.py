import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import kagglehub

path = kagglehub.dataset_download("stealthtechnologies/regression-dataset-for-household-income-analysis")

# --- Auto-detect CSV file ---
files = [f for f in os.listdir(path) if f.endswith(".csv")]
if len(files) == 0:
    raise FileNotFoundError("No CSV file found in dataset folder!")
csv_file = files[0]
print(f"Using file: {csv_file}")

df = pd.read_csv(os.path.join(path, csv_file))
print(df.head())
print(df.info())

# --- Preprocessing ---
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

# --- Features  ---(relation between data is not linear so i add columns i modify data to not be linear like age^2 or log(1+eductaion))
# 3matn data lesa b43a msh 3arfa a7sn wad3 eazy
if "Age" in df.columns: 
    df["Age_squared"] = df["Age"] ** 2
if "Education_Level" in df.columns:
    df["Edu_log"] = np.log1p(df["Education_Level"])
if "Age" in df.columns and "Education_Level" in df.columns:
    df["Age_x_Edu"] = df["Age"] * df["Education_Level"]

X = df.drop("Income", axis=1)
y = df["Income"]

# --- Choose Model ---
use_linear = False  # 1->lineat 0->desicion tree

if use_linear:
    model = LinearRegression()
    model_name = "Linear Regression"
else:
    model = DecisionTreeRegressor(max_depth=6, random_state=42)
    model_name = "Decision Tree Regressor"

# --- Cross Validation (5 folds) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
mae_scores = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")

print(f"\n--- {model_name} Cross-Validation Results (5-fold) ---\n")
print(f"Average R²   : {r2_scores.mean():.4f} (+/- {r2_scores.std():.4f})")
print(f"Average RMSE : {rmse_scores.mean():.4f}")
print(f"Average MAE  : {mae_scores.mean():.4f}")

# ---  plotting actual vs predicted ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train & Predict ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Stats ---
print(df.describe())
print(df.corr()["Income"].sort_values(ascending=False))

# ---Accuracy in regression = R² ---
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n--- {model_name} Holdout Test Results ---\n")
print(f"R² Score (Accuracy) : {r2:.4f}")
print(f"RMSE                : {rmse:.4f}")
print(f"MAE                 : {mae:.4f}")

# --- Plot actual vs predicted ---
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title(f"{model_name}: Actual vs Predicted Income")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()
