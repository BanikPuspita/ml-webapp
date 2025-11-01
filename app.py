# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
MODEL_PATH = "model.pkl"
VECT_PATH = "vectorizer.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(VECT_PATH):
    raise FileNotFoundError("model.pkl or vectorizer.pkl not found. Run model.py first.")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return jsonify({"prediction": pred})

if __name__ == "__main__":
    app.run(debug=True)
