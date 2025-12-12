import os
import sys
import pickle
import importlib.util

# === Load preprocess.py dynamically ===

current_dir = os.path.dirname(os.path.abspath(__file__))
preprocess_path = os.path.join(current_dir, "preprocess.py")

spec = importlib.util.spec_from_file_location("preprocess", preprocess_path)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

clean_text = preprocess.clean_text

# === Load model and TF-IDF ===

BASE_DIR = os.path.dirname(current_dir)
model_path = os.path.join(BASE_DIR, "models", "spam_model.pkl")
vector_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
tfidf = pickle.load(open(vector_path, "rb"))

def predict_sms(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    return model.predict(vector)[0]

if __name__ == "__main__":
    sms = input("Enter SMS text: ")
    print("Prediction:", predict_sms(sms))
