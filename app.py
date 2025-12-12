from flask import Flask, render_template, request
import pickle
import os
print("Template folder:", os.path.join(os.path.dirname(__file__), "templates"))
from src.preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "models", "spam_model.pkl"), "rb"))
tfidf = pickle.load(open(os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl"), "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0].strip().lower()


    return render_template("index.html", result=prediction)

if __name__ == "__main__":
    app.run(debug=True)
