import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import kagglehub

# --- Download dataset  ---
path = kagglehub.dataset_download("uciml/sms-spam-collection-dataset")

# --- Auto-detect CSV ---
files = [f for f in os.listdir(path) if f.endswith(".csv") or f.endswith(".tsv") or f.endswith(".txt")]
if len(files) == 0:
    raise FileNotFoundError("No file found in dataset folder!")
csv_file = files[0]
print(f"Using file: {csv_file}")


df = pd.read_csv(os.path.join(path, csv_file), encoding="latin-1")

# Keep only useful columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

print(df.head())
print(df.info())

# --- Preprocessing ---
df["label"] = df["label"].map({"ham": 0, "spam": 1})  # convert to 0/1

# extra columns
df["msg_length"] = df["message"].apply(len)
df["word_count"] = df["message"].apply(lambda x: len(x.split()))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(stop_words="english", max_features=3000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(df["message"])
X = np.hstack([X_tfidf.toarray(), df[["msg_length", "word_count"]].values])
y = df["label"]

# --- Choose Model ---
use_logistic = True   # 1-> Logistic Regression, 0-> Decision Tree

if use_logistic:
    model = LogisticRegression(max_iter=1000, random_state=42)
    model_name = "Logistic Regression"
else:
    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model_name = "Decision Tree Classifier"

# --- Cross Validation (5 folds) ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_acc = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
cv_f1 = cross_val_score(model, X, y, cv=kf, scoring="f1")

print(f"\n--- {model_name} Cross-Validation Results (5-fold) ---\n")
print(f"Average Accuracy : {cv_acc.mean():.4f} (+/- {cv_acc.std():.4f})")
print(f"Average F1-score : {cv_f1.mean():.4f}")

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train & Predict ---
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Evaluation ---
print(f"\n--- {model_name} Holdout Test Results ---\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title(f"{model_name} - Confusion Matrix")
plt.show()
