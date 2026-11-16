from pathlib import Path
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "tickets.csv")
df = df.drop_duplicates(subset=["ticket_description", "category"]).dropna(subset=["ticket_description", "category"])
df["clean"] = df["ticket_description"].apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["category"], test_size=0.2, random_state=42, stratify=df["category"])

encoder = SentenceTransformer("all-MiniLM-L6-v2")
Xtr = encoder.encode(X_train.tolist(), batch_size=64, show_progress_bar=True)
Xte = encoder.encode(X_test.tolist(), batch_size=64)

clf = LogisticRegression(max_iter=3000, C=10, class_weight="balanced").fit(Xtr, y_train)
pred = clf.predict(Xte)
print("Train accuracy:", round(accuracy_score(y_train, clf.predict(Xtr)), 3))
print("Test accuracy :", round(accuracy_score(y_test, pred), 3))
print(classification_report(y_test, pred, zero_division=0))

(ROOT / "model").mkdir(exist_ok=True)
joblib.dump(clf, ROOT / "model" / "emb_clf.pkl")
print("Model saved.")