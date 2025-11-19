import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

file_path = file_path = r"C:\Users\Avinash R\OneDrive\Desktop\Modetest1\sentiment_dataset.csv"
df = pd.read_csv(file_path, encoding="utf-8")

# ---------------- Load Dataset ----------------

df = pd.read_csv(file_path, encoding="utf-8")

# Fix column names (your CSV uses lowercase)
df['sentiment'] = df['sentiment'].astype(int)

# ---------------- TF-IDF Vectorizer ----------------

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(df['comment'])
y = df['sentiment']

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# Models
# -------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": SVC(probability=True),
    "Random Forest": RandomForestClassifier(class_weight="balanced")
}

best_model = None
best_accuracy = 0

# -------------------------------------------------------
# Training Loop
# -------------------------------------------------------

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"{name} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model

# -------------------------------------------------------
# Save Model + Vectorizer
# -------------------------------------------------------

joblib.dump(best_model, "sentiment_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print(f"\n🎯 Best Model: {best_model.__class__.__name__} | Accuracy: {best_accuracy:.2f}")
print("✔ Model and vectorizer saved successfully!")