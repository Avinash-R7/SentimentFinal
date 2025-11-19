from predict import predict_sentiment

print("\n=== Sentiment Analysis Demo ===")

while True:
    text = input("\nEnter a comment (or type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    result = predict_sentiment(text)

    print("\n--- Result ---")
    print(f"Sentiment       : {result['sentiment']}")
    print(f"Sentiment Value : {result['sentiment_value']}")
    print(f"Confidence      : {result['confidence']}")
