# Project 6 - Sentiment Analysis on Social Media Data 
import os
import pandas as pd
import numpy as np
from textblob import TextBlob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


os.makedirs("data", exist_ok=True)
os.makedirs("output", exist_ok=True)


tweets = [
    "I love this new phone! The camera quality is amazing 😍",
    "This product is terrible. Waste of money.",
    "Honestly, the service was okay. Nothing special.",
    "I am extremely happy with the results!",
    "Worst experience ever, I am disappointed.",
    "The movie was wonderful and full of emotion.",
    "It's fine, not too good, not too bad.",
    "I hate how slow this app is!",
    "Absolutely fantastic customer support!",
    "Feeling neutral about this update."
]

df = pd.DataFrame({"tweet": tweets})
df.to_csv("data/raw_tweets.csv", index=False)

print("✅ Sample tweets created.")
print(df.head())


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


df["sentiment"] = df["tweet"].apply(analyze_sentiment)

print("\n✅ Sentiment analysis complete:")
print(df)


df.to_csv("output/tweets_with_sentiment.csv", index=False)
print("\n✅ Output saved to output/tweets_with_sentiment.csv")


sentiment_counts = df["sentiment"].value_counts()

plt.figure(figsize=(8, 4))
plt.bar(sentiment_counts.index, sentiment_counts.values)
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.grid(True)
plt.show(block=True)


plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%")
plt.title("Sentiment Breakdown")
plt.show(block=True)

input("\nPress ENTER to exit…")
