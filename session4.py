import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("study_data.csv")
print("Dataset:\n", data)

# -------------------- Linear Regression -----------------------
print("\nLinear Regression")

X = data[["hours"]]   
y = data[["score"]]   

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nPredictions on test set:")
for i in range(len(X_test)):
    print(f"Hours studied: {X_test.iloc[i,0]} → Predicted Score: {y_pred[i][0]:.2f}, Actual Score: {y_test.iloc[i,0]}")

example_hours = np.array([[5]])
predicted_score = model.predict(example_hours)
print(f"\nIf a student studies 5 hours → Predicted Score = {predicted_score[0][0]:.2f}")
