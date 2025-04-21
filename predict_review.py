from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
model_path = "./sentiment_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Label mapping
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Prediction function
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits).item()
    return id2label[predicted_class_id]

# Test loop
while True:
    review = input("\nEnter a customer review (or type 'exit' to quit): ")
    if review.lower() == 'exit':
        break
    prediction = predict_sentiment(review)
    print(f"Predicted Sentiment: {prediction}")
