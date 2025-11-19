import whisper
from langdetect import detect
from deep_translator import GoogleTranslator
import joblib

# Load speech-to-text model
whisper_model = whisper.load_model("medium")

# Load sentiment model + vectorizer
sentiment_model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------------------
# Speech to Text
# -------------------
def speech_to_text(audio_file):
    result = whisper_model.transcribe(audio_file)
    return result["text"]

# -------------------
# Language Detection
# -------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# -------------------
# Translation using deep-translator
# -------------------
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

# -------------------
# Sentiment Prediction
# -------------------
def predict_sentiment(text):
    X = vectorizer.transform([text])
    prediction = sentiment_model.predict(X)[0]
    probability = sentiment_model.predict_proba(X)[0].max()
    return prediction, probability

# -------------------
# Full Pipeline
# -------------------
def process_audio(audio_file):
    # Step 1: Speech → Text
    text = speech_to_text(audio_file)
    print(f"\nTranscribed Text: {text}")

    # Step 2: Detect Lang
    lang = detect_language(text)
    print(f"Detected Language: {lang}")

    # Step 3: Tamil or Mixed → Translate
    if lang == "en":
        final_text = text
    elif lang in ["ta", "hi", "kn", "te", "ml"]:   # any Indian lang
        final_text = translate_to_english(text)
        print(f"\nTranslated to English: {final_text}")
    else:
        final_text = text

    # Step 4: Sentiment Analysis
    sentiment, confidence = predict_sentiment(final_text)

    return final_text, sentiment, confidence


# --------------------------
# Usage Example
# --------------------------
audio_path = "positive.mp3"

final_text, sentiment, confidence = process_audio(audio_path)

print("\n--- RESULT ---")
print("English Text:", final_text)
print("Sentiment Value:", sentiment)
print("Confidence:", confidence)
