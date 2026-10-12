"""
Codveda Data Analytics Internship - Level 3 (Advanced)
Task 3: Natural Language Processing (NLP) - Sentiment Analysis
Dataset: Sentiment dataset.csv (social media posts)
"""
import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

RAW_PATH = "/home/claude/dataset1/Data Set For Task/3) Sentiment dataset.csv"
PLOT_DIR = "/home/claude/work/plots"

df = pd.read_csv(RAW_PATH)
df = df[["Text"]].copy()
df["Text"] = df["Text"].astype(str)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)                 # remove punctuation/numbers
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return " ".join(tokens)

# 1. Preprocess text data (tokenization, stopword removal, lemmatization)
df["clean_text"] = df["Text"].apply(preprocess_text)

# 2. Sentiment analysis using TextBlob (polarity-based classification)
def classify_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"

df["polarity"] = df["clean_text"].apply(lambda t: TextBlob(t).sentiment.polarity)
df["sentiment"] = df["clean_text"].apply(classify_sentiment)

print("Sentiment distribution:\n", df["sentiment"].value_counts())
print("\nSample results:\n", df[["Text", "sentiment", "polarity"]].head(10))

df.to_csv("/home/claude/work/level3/sentiment_results.csv", index=False)

# 3. Visualize sentiment distribution
plt.figure(figsize=(6, 5))
order = ["Positive", "Neutral", "Negative"]
sns.countplot(data=df, x="sentiment", order=order, palette=["#2E86AB", "#F4A261", "#E63946"])
plt.title("Sentiment Distribution")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t3_sentiment_distribution.png", dpi=120)
plt.close()

# Polarity histogram
plt.figure(figsize=(7, 5))
sns.histplot(df["polarity"], bins=30, kde=True, color="#A23B72")
plt.title("Polarity Score Distribution")
plt.xlabel("Polarity (-1 = negative, +1 = positive)")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t3_polarity_hist.png", dpi=120)
plt.close()

# 4. Word frequencies using word clouds (overall + by sentiment)
all_text = " ".join(df["clean_text"])
wc_all = WordCloud(width=900, height=500, background_color="white", colormap="viridis").generate(all_text)
plt.figure(figsize=(10, 6))
plt.imshow(wc_all, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud - All Posts")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t3_wordcloud_all.png", dpi=120)
plt.close()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, sentiment, color in zip(axes, ["Positive", "Negative"], ["Greens", "Reds"]):
    text_subset = " ".join(df.loc[df["sentiment"] == sentiment, "clean_text"])
    if text_subset.strip():
        wc = WordCloud(width=700, height=500, background_color="white", colormap=color).generate(text_subset)
        ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Word Cloud - {sentiment} Posts")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/l3_t3_wordcloud_by_sentiment.png", dpi=120)
plt.close()

print("\nNLP sentiment analysis complete. Plots saved to", PLOT_DIR)
