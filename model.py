# model.py
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def train_and_save():
    # Very small sample data for demo — replace with your dataset later
    texts = [
        "I love this!", "This is bad", "I feel amazing", 
        "It was terrible", "I like it", "Not good at all"
    ]
    labels = ["positive", "negative", "positive", "negative", "positive", "negative"]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = MultinomialNB()
    model.fit(X, labels)

    # Save model and vectorizer
    joblib.dump(model, "model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")
    print("Saved model.pkl and vectorizer.pkl")

if __name__ == "__main__":
    train_and_save()
