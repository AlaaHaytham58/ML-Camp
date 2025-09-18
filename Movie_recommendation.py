# ---------------- IMPORTS -----------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.decomposition import PCA

# ---------------- LOAD DATA -----------------
df = pd.read_csv("movies.csv")
required_columns = ["genres", "keywords", "overview", "title"]
#drop null rows
df = df[required_columns].dropna().reset_index(drop=True)

# ---------------- WORD CLOUD -----------------
df['combined'] = df['genres'] + " " + df['keywords'] + " " + df['overview']
combined_text = " ".join(df['combined'])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Most Common Words in Movie Dataset")
plt.show()

# ---------------- TEXT CLEANING -----------------
nltk.download("punkt")
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["cleaned_text"] = df["combined"].apply(preprocess_text)

# ---------------- TF-IDF VECTORIZE -----------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["cleaned_text"])

# ---------------- GENRE LABEL ENCODING -----------------
# take only the first genre (if multiple)
df["main_genre"] = df["genres"].apply(lambda x: x.split("|")[0] if "|" in x else x)
le = LabelEncoder()
y = le.fit_transform(df["main_genre"])

# ---------------- TRAIN / TEST SPLIT -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- KNN  -----------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))
labels_used = np.unique(y_test)
print(classification_report(y_test, y_pred_knn, labels=labels_used, target_names=le.classes_[labels_used]))

# ---------------- DECISION TREE  -----------------
dt = DecisionTreeClassifier(max_depth=20, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print(" Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
labels_used = np.unique(y_test)
print(classification_report(y_test, y_pred_dt, labels=labels_used, target_names=le.classes_[labels_used]))

# ---------------- MOVIE RECOMMENDER  -----------------
def recommend_movies(movie_title, num_recommendations=5):
    if movie_title not in df["title"].values:
        print(f" Movie '{movie_title}' not found in dataset.")
        return
    idx = df[df["title"] == movie_title].index[0]
    cosine_sim = cosine_similarity(X[idx], X).flatten()
    similar_indices = cosine_sim.argsort()[-num_recommendations-1:-1][::-1]
    print(f"\n Recommended movies for '{movie_title}':")
    print(df.iloc[similar_indices]["title"].values)

recommend_movies("Titanic")

recommend_movies("Captain Phillips")