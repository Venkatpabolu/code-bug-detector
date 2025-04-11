import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
df = pd.read_csv("code_dataset.csv")

# Clean code column
df["code"] = df["code"].str.replace(r"[^a-zA-Z0-9\s]", " ", regex=True).str.lower()

# Split
X_train, X_test, y_train, y_test = train_test_split(df["code"], df["label"], test_size=0.2, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Save model and vectorizer
joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
print("✅ Model and Vectorizer saved successfully!")
