import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer

# Load your dataset
df = pd.read_csv("business_reviews.csv")

# Label mapping
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
df['label'] = df['label'].map(label2id)

# Convert to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(example):
    return tokenizer(example["review"], padding="max_length", truncation=True)

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function)

# Print a sample for verification
print(tokenized_dataset[0])
