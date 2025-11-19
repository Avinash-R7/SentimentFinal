import whisper
import joblib

def predict_from_audio(audio_path):

    # Step 1: Load Whisper (Tamil + English)
    whisper_model = whisper.load_model("large")
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_path, fp16=False)
    text = result["text"]
    print("Transcription:", text)

    # Step 2: Load your trained model
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    # Step 3: Predict sentiment
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = max(model.predict_proba(X)[0])

    return text, prediction, confidence


# ------------ RUNNING THE FUNCTION ------------
if __name__ == "__main__":
    file_path = "voice1.wav"  # change to your file name
    text, sentiment, conf = predict_from_audio(file_path)
    print("\nFinal Output:")
    print("Transcribed Text:", text)
    print("Sentiment:", sentiment)
    print("Confidence:", conf)
