import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle
import os
from preprocess import clean_text

print("TRAINING STARTED...")

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(BASE_DIR, "data", "spam.csv")

print("Loading:", data_path)

if not os.path.exists(data_path):
    print("ERROR: Dataset not found!")
    exit()

df = pd.read_csv(data_path, encoding="latin-1")
df = df[['v1','v2']]
df.columns = ['label','message']

df['cleaned'] = df['message'].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['label'], test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

pred = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, pred))

models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

pickle.dump(model, open(os.path.join(models_dir, "spam_model.pkl"), "wb"))
pickle.dump(tfidf, open(os.path.join(models_dir, "tfidf_vectorizer.pkl"), "wb"))

print("MODEL TRAINED AND SAVED!")
