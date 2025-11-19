import joblib
import numpy as np

# Load model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_sentiment(text):
    X = vectorizer.transform([text])
    
    pred = model.predict(X)[0]
    
    try:
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
    except:
        confidence = None
    
    label_map = {
        1: "Positive",
        0: "Neutral",
        -1: "Negative"
    }

    return {
        "sentiment_value": int(pred),
        "sentiment": label_map.get(pred, "Unknown"),
        "confidence": confidence
    }
