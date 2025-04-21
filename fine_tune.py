from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from datasets import Dataset
import pandas as pd

# Load dataset
df = pd.read_csv("business_reviews.csv")
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label'].map(label2id)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Tokenize function
def tokenize_function(example):
    return tokenizer(example["review"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function)

# Set training args
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# Train!
trainer.train()

# Save model
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

print("✅ Model trained and saved to /sentiment_model")
