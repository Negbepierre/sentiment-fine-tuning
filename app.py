import gradio as gr
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Load fine-tuned model
model = BertForSequenceClassification.from_pretrained("sentiment_model")
tokenizer = BertTokenizer.from_pretrained("sentiment_model")

id2label = {0: "negative", 1: "neutral", 2: "positive"}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits).item()
    return id2label[predicted_class]

iface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Enter a review..."),
    outputs=gr.Text(label="Predicted Sentiment"),
    title="Sentiment Classifier",
    description="Enter a customer review to get its sentiment using BERT."
)

iface.launch()
