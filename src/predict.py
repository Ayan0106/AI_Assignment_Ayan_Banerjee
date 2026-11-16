from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer
from preprocessing import clean_text

ROOT = Path(__file__).resolve().parent.parent
clf = joblib.load(ROOT / "model" / "emb_clf.pkl")
encoder = SentenceTransformer("all-MiniLM-L6-v2")

def predict_ticket(text, threshold=40):
    p = clf.predict_proba(encoder.encode([clean_text(text)]))[0]
    i = p.argmax()
    cat, conf = clf.classes_[i], p[i] * 100
    return ("Needs human review", conf) if conf < threshold else (cat, conf)

if __name__ == "__main__":
    print("=" * 36, "\n Customer Support Classifier\n", "=" * 36)
    text = input("Enter Ticket Description:\n> ")
    cat, conf = predict_ticket(text)
    print(f"\nPredicted Category: {cat}\nConfidence: {conf:.1f}%")