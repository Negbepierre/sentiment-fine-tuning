import pandas as pd
import random

positive_reviews = [
    "Great service, I’ll definitely come back!",
    "The tailoring was perfect, thanks a lot!",
    "Loved the meal, so tasty and fresh!",
    "Very professional barber, great experience.",
    "Fast delivery and quality product."
]

neutral_reviews = [
    "It was okay, nothing special.",
    "The experience was average.",
    "Not bad, not great either.",
    "Service was fine, nothing more.",
    "It worked, but it could be better."
]

negative_reviews = [
    "Terrible customer service!",
    "My clothes were ruined.",
    "Haircut was uneven and rushed.",
    "Food was cold and had no taste.",
    "Delivery was late and the product was damaged."
]

data = []

for _ in range(100):
    label = random.choice(['positive', 'neutral', 'negative'])
    if label == 'positive':
        text = random.choice(positive_reviews)
    elif label == 'neutral':
        text = random.choice(neutral_reviews)
    else:
        text = random.choice(negative_reviews)
    data.append({'review': text, 'label': label})

df = pd.DataFrame(data)
df.to_csv("business_reviews.csv", index=False)
print("✅ Dataset saved as 'business_reviews.csv'")
