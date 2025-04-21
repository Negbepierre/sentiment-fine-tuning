# 🧠 Sentiment Fine-Tuning with BERT

This project demonstrates how to fine-tune a pre-trained BERT model on custom business review data using Hugging Face Transformers and PyTorch.

---

## 📌 Project Overview

I created a simple AI model that classifies customer reviews as:
- Positive 😊
- Neutral 😐
- Negative 😠

---

## 🛠 What I Did

1. **Generated a custom dataset** of synthetic business reviews with sentiment labels
2. **Preprocessed and tokenized** the data using `BertTokenizer`
3. **Fine-tuned** the `bert-base-uncased` model using Hugging Face’s `Trainer`
4. **Saved and tested** the trained model using real review text

---

## 📁 Folder Structure

sentiment-fine-tuning/ ├── generate_reviews.py # Generate synthetic reviews ├── preprocess_reviews.py # Preprocess and tokenize data ├── fine_tune.py # Fine-tune the BERT model ├── predict_review.py # Test the trained model ├── sentiment_model/ # Saved model and tokenizer ├── business_reviews.csv # Dataset (optional, .gitignore) └── README.md # Project explanation


---

## ✅ Example Usage

```bash
$ python predict_review.py
Enter a customer review: Too slow and it lags
Predicted Sentiment: negative

Tools Used

Python

Hugging Face Transformers

PyTorch

Pandas

Datasets

