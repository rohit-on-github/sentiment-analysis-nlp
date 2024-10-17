import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the English NLP model from spaCy
nlp = spacy.load('en_core_web_sm')

# Example data: Replace with your own dataset (CSV file or scraped data)
data = {
    'review': [
        "I loved this product! It works really well.",
        "This is the worst product I’ve ever bought.",
        "I am extremely happy with my purchase.",
        "The quality is bad, not worth the money.",
        "Amazing experience, highly recommend!",
        "Terrible customer service, very disappointed."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0]  # 1: Positive, 0: Negative
}

df = pd.DataFrame(data)

# Preprocessing: Tokenization, Lemmatization, Removing stopwords and punctuation
def preprocess_text(text):
    doc = nlp(text.lower())  # Convert to lowercase and tokenize with spaCy
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

df['cleaned_review'] = df['review'].apply(preprocess_text)

# Splitting the data into training and test sets
X = df['cleaned_review']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Predictions
y_pred = model.predict(X_test_tfidf)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

