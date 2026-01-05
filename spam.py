import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load Kaggle dataset
df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep required columns
df = df[["v1", "v2"]]
df.columns = ["label", "message"]

# Convert labels to numbers
df["label"] = df["label"].map({"spam": 1, "ham": 0})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["message"], df["label"], test_size=0.2, random_state=42
)

# Text vectorization (TF-IDF)
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Predictions
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test with a new message
new_msg = ["Congratulations! You have won a free voucher"]
new_msg_vec = vectorizer.transform(new_msg)
prediction = model.predict(new_msg_vec)

print("\nMessage is:", "Spam" if prediction[0] == 1 else "Not Spam")
