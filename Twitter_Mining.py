# Importing Dependencies

# Data Manipulation
import pandas as pd
import numpy as np
from collections import Counter

# Text Preprocessing
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Visualization & Plotting
import matplotlib.pyplot as plt


tweets_df = pd.read_json("data/raw/nikelululemonadidas_tweets.jsonl", lines=True)

# Display basic info and properties of the dataframe
tweets_df.info()
print(f"\nFirst 5 rows of the 'full_text' column:\n{tweets_df['full_text'].head()}")
print(f"\nFirst row of the dataframe:\n{tweets_df.iloc[0]}\n")

# Extract attributes like id, created_at, retweet_count, text from the tweets and create a new dataframe with these attributes
tweets_subset_df = tweets_df[['id_str', 'created_at', 'retweet_count', 'full_text', 'entities']]

# Remove duplicates & Empty Rows
tweets_subset_df.drop_duplicates(subset=['id_str', 'full_text', 'created_at'], inplace=True) # Exclude entities
tweets_subset_df = tweets_subset_df.dropna()

# Load the English stopword list
stopword_set = set(stopwords.words('english') + ['rt', 'amp', 'via', 'u', 'im', 'let'])

# Discard negation terms from the stopword list to preserve their meaning in sentiment analysis
stopword_set.discard('not')
stopword_set.discard('no')

# Text Precprocessing
def preprocess(text, tokenizer=TweetTokenizer(), lemmatizer=WordNetLemmatizer(), stopword_set=stopword_set):
    text = text.lower() # Casefolding
    text = re.sub(r"http\S+", "", text) # Remove URLs
    text = re.sub(r"@\w+", "", text) # remove mentions
    text = re.sub(r'[^a-zA-Z#\s]', '', text) # Remove special characters, punctuation, and numbers (but keep hashtags)
    tokens = tokenizer.tokenize(text) # Tokenization
    tokens = [lemmatizer.lemmatize(token) for token in tokens] # Lemmatization
    tokens = [token for token in tokens if token not in stopword_set] # Remove stopwords
    
    return tokens

# Apply the preprocessing function to the 'full_text' column of the tweets dataframe and create a new column to store the preprocessed tokens
tweets_subset_df['tokens'] = tweets_subset_df['full_text'].apply(preprocess)


# Term Frequency Analysis
term_frequency = Counter()

# Iterate through the preprocessed tokens in the 'tokens' column of the tweets dataframe and update the term frequency counter
for tokens in tweets_subset_df['tokens']:
    term_frequency.update(tokens)

# Display the 10 most common terms and their frequencies
for t,x in term_frequency.most_common(10):
    print('Term: {} - Frequency: {}'.format(t, x))


# Entity Analysis
hashtag_frequency = Counter()

# Define a hashtag extracting function
def get_hashtags(entities):
    if isinstance(entities, dict): # Check for NaN values by verifying whether the 'entities' field is a dictionary
        hashtags = entities.get('hashtags', [])

        tags = [tag['text'].lower() for tag in hashtags if 'text' in tag]
        hashtag_frequency.update(tags)

tweets_subset_df['entities'].apply(get_hashtags)

# Display the 10 most common hashtags and their frequencies
for tag, count in hashtag_frequency.most_common(10):
    print(f"Hashtag: #{tag} - Frequency: {count}")


# Extract Mentions
def get_mentions(entities):
    if isinstance(entities, dict):
        return [mention['screen_name'].lower() for mention in entities.get('user_mentions', [])]
    return []

tweets_subset_df['mentions'] = tweets_subset_df['entities'].apply(get_mentions)

mention_frequency = Counter()

for mentions in tweets_subset_df['mentions']:
    mention_frequency.update(mentions)

for user, count in mention_frequency.most_common(10):
    print(f"Mention: @{user} - Frequency: {count}")


# Define a function to detect and extract brand mentions in the tweet text
def get_brand(tweet):
    casefolded_text = str(tweet['full_text']).lower()

    brands = []

    if re.search(r'\bnike\b', casefolded_text): # Using word boundaries to avoid partial matches (e.g., "nike" in "nikeair")
        brands.append('Nike')
    if re.search(r'\blululemon\b', casefolded_text):
        brands.append('Lululemon')
    if re.search(r'\badidas\b', casefolded_text):
        brands.append('Adidas')

    return brands if brands else ['Other']

tweets_subset_df['brand'] = tweets_subset_df.apply(get_brand, axis=1)

# Display the first 10 rows of the 'full_text' and 'brand' columns to verify the brand detection results
print(tweets_subset_df[['full_text', 'brand']].head(10))