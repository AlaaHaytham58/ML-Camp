import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,mean_squared_error,r2_score

import kagglehub

train_path = "/kaggle/input/loan-predication/train_u6lujuX_CVtuZ9i (1).csv"

data = pd.read_csv(train_path)
print(" Dataset loan:", data.shape)
print(data.head())
print("\nMissing values:\n", data.isnull().sum())

#-------------------preprocessing for data -----------------------------


# i use fields i cant get it median by mode
for col in ["Gender", "Married", "Dependents", "Self_Employed", "Credit_History"]:
    data[col].fillna(data[col].mode()[0], inplace=True)

#i use median for numerical data
for col in ["LoanAmount", "Loan_Amount_Term"]:
    data[col].fillna(data[col].median(), inplace=True)

# Encode  variables
le = LabelEncoder()
for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    data[col] = le.fit_transform(data[col])

print("\nData after preprocessing:\n", data.head())

#--------------------------DONE PREPROCESSING-------------------------

#_____________________________________________________________________


#_____________________________LINEAR REGRESSION_______________________

X_lin = data[["ApplicantIncome"]]
y_lin = data[["LoanAmount"]]

X_train, X_test, y_train, y_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

print("\n--- Linear Regression ---")
print("\n--- Linear Regression Test Results ---")
for i in range(10):
    print(f"Applicant Income: {X_test.iloc[i,0]} → "
          f"Predicted LoanAmount: {y_pred_lin[i][0]:.2f}, "
          f"Actual LoanAmount: {y_test.iloc[i,0]}")
print("MSE:", mean_squared_error(y_test, y_pred_lin))
print("R2 Score:", r2_score(y_test, y_pred_lin))

#-----------------------DONE LINEAR REGRESSION------------------------

#_____________________________________________________________________

#________________________MULTIPLE REGRESSION__________________________

X_multi = data[["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term", "Credit_History"]]
y_multi = data[["LoanAmount"]]

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

multi_model = LinearRegression()
multi_model.fit(X_train_m, y_train_m)
y_pred_multi = multi_model.predict(X_test_m)

print("\n--- Multiple Regression ---")
for i in range(10):
    print(f"Applicant Income: {X_test_m.iloc[i,0]} → "
          f"Predicted LoanAmount: {y_pred_multi[i][0]:.2f}, "
          f"Actual LoanAmount: {y_test_m.iloc[i,0]}")
print("MSE:", mean_squared_error(y_test_m, y_pred_multi))
print("R2 Score:", r2_score(y_test_m, y_pred_multi))
#____________________DONE MULTIPLE REGRESSION__________________________
X_log = data.drop(columns=["Loan_ID", "Loan_Status"])
y_log = data["Loan_Status"]

# Scale features
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_scaled, y_log, test_size=0.2, random_state=42)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train_log, y_train_log)
y_pred_log = log_model.predict(X_test_log)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test_log, y_pred_log))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_log, y_pred_log))
print("\nClassification Report:\n", classification_report(y_test_log, y_pred_log))
